"""Keep experimental scopes separate and preserve raw evidence in reports."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ditserve.evaluation import workflow_report
from ditserve.evaluation.workflow_report import build_report, distribution, render_html


class WorkflowReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "run01"
        self.root.mkdir()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_final_report_reuses_pruned_archive_without_rediscovery(self):
        saved = dict(schema="ditserve.run.v1", created_utc="original", metrics={"stable": 1})
        with mock.patch.object(workflow_report, "archived_run", return_value=saved), \
             mock.patch.object(workflow_report, "discover", side_effect=AssertionError("must not rediscover pruned inputs")):
            self.assertEqual(build_report(self.root), saved)

    def test_preview_never_finalizes_or_changes_canonical_files(self):
        canonical = self.write("run.json", dict(schema="ditserve.run.v1", metrics={"stable": 1}))
        before = canonical.read_bytes()
        analysis = self.root / "ANALYSIS.html"
        analysis.write_text("retained final report")
        with mock.patch.object(workflow_report, "finalize_workflow") as finalize:
            workflow_report.main(["--root", str(self.root), "--preview"])
        finalize.assert_not_called()
        self.assertEqual(canonical.read_bytes(), before)
        self.assertEqual(analysis.read_text(), "retained final report")
        self.assertTrue((self.root / "ANALYSIS-LIVE.html").is_file())

    def test_final_report_preserves_existing_pending_dependencies(self):
        artifact = dict(path="quality.npy", bytes=1, sha256="a" * 64, kind="raw",
                        dependencies=[dict(closed=False, reason="blind review pending")])
        self.write("run.json", dict(schema="ditserve.run.v1", retention=dict(pending_dependencies=["quality"]),
                                   artifacts=[artifact]))
        run = build_report(self.root)
        self.assertEqual(run["retention"]["pending_dependencies"], ["quality"])
        self.assertIn(artifact, run["artifacts"])

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl is required for archive round-trip")
    def test_keep_raw_then_prune_and_repeat_preserves_published_metrics(self):
        capture = b"retained profiler capture"
        self.write("traces/sample.parsed.json", dict(schema_version="ditserve.workflow_trace.v1", status="completed",
            source=str(self.root / "traces/sample.sqlite"), postprocess=dict(capture=dict(
                bytes=len(capture), sha256=hashlib.sha256(capture).hexdigest())), arms=[]))
        (self.root / "traces/sample.nsys-rep").write_bytes(capture)
        sqlite = self.root / "traces/sample.sqlite"
        sqlite.write_bytes(b"redundant export")
        destination = Path(self.directory.name) / "archive"
        args = ["--root", str(self.root), "--destination", str(destination)]
        workflow_report.main(args + ["--keep-raw"])
        before = json.loads((self.root / "run.json").read_text())
        html_before = (self.root / "ANALYSIS.html").read_bytes()
        self.assertTrue(sqlite.exists())
        workflow_report.main(args)
        self.assertFalse(sqlite.exists())
        current = json.loads((self.root / "run.json").read_text())
        self.assertEqual(current["created_utc"], before["created_utc"])
        self.assertEqual(current["metrics"], before["metrics"])
        canonical = (self.root / "run.json").read_bytes()
        workflow_report.main(args)
        self.assertEqual((self.root / "run.json").read_bytes(), canonical)
        self.assertEqual((self.root / "ANALYSIS.html").read_bytes(), html_before)

    def test_trace_link_uses_retained_capture(self):
        self.write("traces/sample.parsed.json", dict(schema_version="ditserve.workflow_trace.v1", status="completed",
            source=str(self.root / "traces/sample.sqlite"), availability=dict(cuda_kernels=True, cuda_memcpy=True),
            arms=[dict(name="sample", host_span_ms=10, devices={"1":dict(compute_union_ms=5, communication={})})]))
        (self.root / "traces/sample.nsys-rep").write_bytes(b"capture")
        page = render_html(build_report(self.root))
        self.assertIn('href="traces/sample.nsys-rep"', page)
        self.assertNotIn('href="traces/sample.sqlite"', page)
        self.assertIn("SQLite 是可重新导出的中间文件", page)

    def encoder(self, wall=3):
        def phase(name, purpose, wall, request="T0-f17-r0"):
            return dict(name=name, status="completed", metadata=dict(variant="T0", purpose=purpose, request_id=request, repeat=0),
                        wall_ms=wall, process_cpu_ms=wall*8, peak_scope="phase", peak_allocated_increment_bytes=100,
                        torch_after=dict(device="cuda:1", peak_allocated_bytes=1000, peak_reserved_bytes=1200, device_total_bytes=2000))
        return dict(schema="ditserve.workflow.encoder.v1", variant="T0", status="completed",
                    phase_records=[phase("text_encoder","video_measurement",wall), phase("text_encoder","video_measurement",wall),
                                   phase("dit_step","video_measurement",5),phase("dit_step","video_measurement",5),
                                   phase("text_encoder","warmup",100), phase("text_encoder","numerical_validation",200)],
                    requests=[dict(kind="video",repeat=0,purpose="video_measurement",frames=17,inference_wall_ms=16),
                              dict(kind="video",repeat=-1,purpose="warmup",frames=17,inference_wall_ms=200)], numeric=[])

    def test_per_request_branches_and_steps_sum_and_contexts_are_excluded(self):
        self.write("encoder-v2/T0/result.json", self.encoder())
        self.write("encoder/T0/result.json", self.encoder(11))
        self.write("preflight/encoder/T0/result.json", self.encoder(13))
        self.write("traces/encoder/T0/result.json", self.encoder(17))
        report = build_report(self.root)
        stages = {row["stage"]: row for row in report["metrics"]["stage_resources"]}
        self.assertEqual(stages["text_encoding"]["wall_ms_per_request"]["median"], 6)
        self.assertEqual(stages["dit_denoising"]["wall_ms_per_request"]["median"], 10)
        self.assertEqual(stages["text_encoding"]["process_average_cores"], 8)
        self.assertEqual(stages["text_encoding"]["peak_allocated_bytes"]["median"], 1000)
        self.assertEqual(stages["text_encoding"]["request_wall_share_percent"]["median"], 37.5)
        self.assertEqual(stages["dit_denoising"]["request_wall_share_percent"]["median"], 62.5)
        self.assertEqual(stages["text_encoding"]["request_wall_denominator_fields"], ["inference_wall_ms"])
        self.assertEqual(len(report["requests"]), 1)
        roles = {row["role"] for row in report["metrics"]["sources"]}
        self.assertEqual(roles, {"performance","superseded_encoder","excluded_context","profiled_workload"})
        self.assertEqual(report["status"], "partial")
        json.dumps(report, allow_nan=False)

    def test_trace_counters_are_weighted_and_not_used_as_latency(self):
        self.write("encoder-v2/T0/result.json",self.encoder())
        phases=[]
        for covered,mean in ((10,20),(30,60)):
            phases.append(dict(name="text_encoder",included=True,metadata=dict(variant="T0",purpose="video_measurement",request_id="T0-f17-r0"),
                               devices={"1":dict(host_window_metrics={"sm_active_percent":dict(mean=mean,covered_ns=covered,window_ns=covered)})}))
        self.write("traces/T0-summary.json",dict(schema_version="ditserve.workflow_trace.v1",status="completed",phases=phases,arms=[]))
        report=build_report(self.root)
        counter=report["metrics"]["stage_hardware_counters"][0]
        self.assertEqual(counter["metrics"]["sm_active_percent"]["mean"],50)
        self.assertEqual(counter["frames"],17)
        self.assertEqual(report["metrics"]["stage_resources"][1]["wall_ms_per_request"]["median"],6)

    def test_paired_ratios_are_computed_from_equal_workloads(self):
        self.write("E1-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E1",status="complete",
            config=dict(frames=17,steps=20,height=480,width=832),pairs=[dict(repeat=0,modes={"serial":dict(wall_ms=100),"prefetch":dict(wall_ms=80)},correctness=[dict(exact=True)])]))
        report=build_report(self.root)
        candidate=next(row for row in report["metrics"]["overlap_experiments"] if row["mode"]=="prefetch")
        self.assertEqual(candidate["paired_speedup"]["median"],1.25)
        self.assertTrue(candidate["numeric_passed"])
        self.assertIsNone(candidate["trace_overlap_proven"])

    def test_e1_namespace_cpu_spans_are_observations_not_summed_request_shares(self):
        def phase(name,wall):
            return dict(name=name,metadata={},status="completed",wall_ms=wall,start_ns=0,end_ns=100000000)
        value=self.overlap_result("E1",17)
        value["phases"]=[phase("E1/r0/serial/request.0/cpu_encode",10),phase("E1/r0/serial/request.1/cpu_encode",12),
                         phase("E1/r0/prefetch/request.0/cpu_encode",15),phase("E1/r0/prefetch/request.1/cpu_encode",20),
                         phase("E1/r0/prefetch/request.1/dit.step.0",70),phase("E1/r-1/prefetch/request.1/cpu_encode",999),
                         phase("warm/cpu_encode",999)]
        self.write("overlap/E1-f17/run.json",value)
        self.write("profiling/E1-f17/run.json",dict(value,status="profiled"))
        report=build_report(self.root)
        rows=report["metrics"]["e1_cpu_encoding"]
        self.assertEqual(len(rows),4)
        later=next(row for row in rows if row["mode"]=="prefetch" and row["request_group"]=="remaining_requests")
        self.assertEqual(later["cpu_encode_wall_ms"]["median"],20)
        self.assertEqual(later["windows"],1)
        self.assertEqual(later["request_indices"],[1])
        self.assertNotIn("request_wall_share_percent",later)
        self.assertEqual(report["metrics"]["stage_resources"],[])
        queued=next(row for row in report["metrics"]["overlap_experiments"] if row["mode"]=="prefetch")
        self.assertEqual(queued["wall_ms"]["median"],80)
        self.assertIn("不证明 CPU 竞争",render_html(report))

    def overlap_job(self,experiment,frames,identifier=None,output=None):
        identifier=identifier or f"{experiment}-f{frames}"
        return dict(id=identifier,command=["python","-m","ditserve.evaluation.workflow_overlap","--experiment",experiment,
            "--frames",str(frames),"--out",f"/data/workflow/run01/{output or 'overlap/'+identifier}"])

    def overlap_result(self,experiment,frames):
        return dict(schema="ditserve.workflow-overlap.v1",experiment=experiment,status="complete",config=dict(frames=frames),
                    pairs=[dict(modes={"serial":dict(wall_ms=100),"prefetch":dict(wall_ms=80)},correctness=True)])

    def test_planned_missing_frame_job_keeps_phase_incomplete(self):
        self.write("jobs.json",dict(jobs=[self.overlap_job("E1",17),self.overlap_job("E1",33)]))
        self.write("overlap/E1-f17/run.json",self.overlap_result("E1",17))
        report=build_report(self.root)
        phase=next(row for row in report["phases"] if row["id"]=="E1")
        self.assertEqual(phase["status"],"incomplete")
        self.assertEqual(phase["metrics"]["expected_normal_jobs"],["E1-f17","E1-f33"])
        counts=report["metrics"]["planned_coverage"]["counts"]["normal"]
        self.assertEqual((counts["expected"],counts["completed"],counts["pending"]),(2,1,1))

    def test_completed_normal_phases_do_not_complete_missing_profiler_and_replay_plan(self):
        for variant in ("T0","T1","T2","T3","T4","T5"):
            self.write(f"encoder-v2/{variant}/result.json",dict(self.encoder(),variant=variant))
        for experiment in ("E1","E2","E3","E4","E5","E6"):
            self.write(f"overlap/{experiment}-f17/run.json",self.overlap_result(experiment,17))
        normal=self.overlap_job("E1",17)
        trace=self.overlap_job("E1",17,"trace-E1",output="profiling/E1")
        trace["command"]=["nsys","profile","--output=/data/workflow/run01/traces/E1"]+trace["command"]
        ncu=dict(id="ncu-T2",command=["ncu","-o","/data/workflow/run01/ncu/T2","python","-m","ditserve.evaluation.workflow_encoder",
                                     "--child","T2","--output","/data/workflow/run01/profiling/T2"])
        replay=dict(id="replay",command=["python","-m","ditserve.evaluation.workflow_embedding_replay","--out","/data/workflow/run01/replay"])
        self.write("jobs.json",dict(jobs=[normal]))
        self.write("jobs-trace.json",dict(jobs=[trace]))
        self.write("jobs-ncu.json",dict(jobs=[ncu]))
        self.write("jobs-replay.json",dict(jobs=[replay]))
        capture=self.root/"traces/E1.nsys-rep";capture.parent.mkdir();capture.write_bytes(b"unparsed capture")
        report=build_report(self.root)
        self.assertTrue(all(row["status"]=="completed" for row in report["phases"]))
        self.assertEqual(report["status"],"partial")
        coverage=report["metrics"]["planned_coverage"]
        self.assertFalse(coverage["all_complete"])
        self.assertEqual(coverage["counts"]["trace"]["capture_available"],1)
        self.assertEqual(coverage["counts"]["trace"]["parsed_available"],0)
        self.assertEqual(coverage["counts"]["ncu"]["pending"],1)
        self.assertEqual(coverage["counts"]["replay"]["pending"],1)
        self.assertIn("计划覆盖",render_html(report))

    def test_plans_deduplicate_ids_exclude_probe_and_honor_superseded_replacement(self):
        old=self.overlap_job("E5",17)
        new=self.overlap_job("E5",17,"E5-f17-v4",output="overlap-v4/E5-f17")
        probe=dict(id="probe",command=["python","-m","ditserve.evaluation.workflow_probe","--out","/data/workflow/run01/probes"])
        self.write("jobs.json",dict(jobs=[old,probe]))
        self.write("jobs-e5.json",dict(jobs=[new]))
        self.write("jobs-copy.json",dict(jobs=[new]))
        self.write("overlap/E5-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E5",status="superseded_before_execution",
                    performance_evidence=False,planned_replacement="E5-f17-v4"))
        self.write("overlap-v4/E5-f17/run.json",self.overlap_result("E5",17))
        report=build_report(self.root)
        coverage=report["metrics"]["planned_coverage"]
        self.assertEqual(len(coverage["jobs"]),2)
        self.assertEqual(coverage["counts"]["normal"]["expected"],1)
        self.assertEqual(coverage["counts"]["normal"]["superseded"],1)
        self.assertEqual(next(row for row in report["phases"] if row["id"]=="E5")["status"],"completed")

    def test_profiler_coverage_requires_capture_and_available_parse(self):
        job=self.overlap_job("E1",17,"trace-E1",output="profiling/E1")
        job["command"]=["nsys","profile","-o","/data/workflow/run01/traces/E1"]+job["command"]
        self.write("jobs-trace.json",dict(jobs=[job]))
        self.write("traces/E1.parsed.json",dict(schema_version="ditserve.workflow_trace.v1",status="completed",
            source="/data/workflow/run01/traces/E1.sqlite",phases=[],arms=[],
            availability=dict(cuda_kernels=True,cuda_memcpy=True,gpu_metrics=True,workflow_nvtx=True)))
        self.assertEqual(build_report(self.root)["metrics"]["planned_coverage"]["counts"]["trace"]["completed"],0)
        (self.root/"traces/E1.nsys-rep").write_bytes(b"original immutable capture")
        coverage=build_report(self.root)["metrics"]["planned_coverage"]
        self.assertEqual(coverage["counts"]["trace"]["completed"],1)
        self.assertEqual(coverage["counts"]["trace"]["parsed_available"],1)
        self.assertTrue(coverage["all_complete"])

    def test_explicit_include_is_scoped_and_oom_is_attempted_not_performance_passed(self):
        path=self.write("encoder-v2/T3/result.json",dict(schema="ditserve.workflow.encoder.v1",variant="T3",status="oom"))
        self.write("jobs.json",dict(jobs=[dict(id="T3",command=["python","-m","ditserve.evaluation.workflow_encoder",
                    "--phases","T3","--output","/data/workflow/run01/encoder-v2"])]))
        report=build_report(self.root)
        phase=next(row for row in report["phases"] if row["id"]=="T3")
        self.assertEqual(phase["status"],"attempted")
        self.assertEqual(phase["qualification"]["scopes"]["performance"]["status"],"unverified")
        self.assertEqual(report["metrics"]["planned_coverage"]["counts"]["normal"]["attempted_negative"],1)
        selected=build_report(self.root,include=[path])
        self.assertEqual(selected["config"]["completion_scope"],"selected_evidence")
        self.assertFalse(selected["metrics"]["planned_coverage"]["enforced"])

    def test_e3_input_setup_scope_and_full_cost_speedup_are_preserved(self):
        self.write("overlap/E3-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E3",status="complete",
            config=dict(frames=17),pairs=[dict(modes={
                "sync":dict(wall_ms=100,input_setup_ms=20,including_input_setup_ms=120,preencoded=True),
                "async":dict(wall_ms=80,input_setup_ms=50,including_input_setup_ms=130,preencoded=True)},correctness=True)]))
        report=build_report(self.root)
        row=next(row for row in report["metrics"]["overlap_experiments"] if row["mode"]=="async")
        self.assertTrue(row["condition"]["preencoded"])
        self.assertEqual(row["input_setup_ms"]["median"],50)
        self.assertEqual(row["including_input_setup_ms"]["median"],130)
        self.assertEqual(row["paired_speedup"]["median"],1.25)
        self.assertEqual(row["paired_speedup_including_input_setup"]["median"],120/130)
        self.assertIn("pinning",render_html(report))

    def test_e2_other_single_uses_own_devices_ratio_and_correctness(self):
        self.write("E2-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E2",status="complete",
            config=dict(frames=17,steps=20),pairs=[dict(repeat=0,preencoded=True,
                single_device="cuda:1",other_single_device="cuda:0",
                modes={"single":dict(wall_ms=100,devices=["cuda:1"]),
                       "single_other":dict(wall_ms=120,devices=["cuda:0"]),
                       "dual":dict(wall_ms=60,devices=["cuda:0","cuda:1"])},
                correctness=[dict(exact=True)],other_single_correctness=[dict(exact=False)],
                speedup=999,other_single_speedup=999)]))
        report=build_report(self.root)
        rows={row["mode"]:row for row in report["metrics"]["overlap_experiments"]}
        self.assertEqual(rows["single_other"]["devices"],["cuda:0"])
        self.assertEqual(rows["single_other"]["paired_speedup"]["median"],100/120)
        self.assertFalse(rows["single_other"]["numeric_passed"])
        self.assertTrue(rows["dual"]["numeric_passed"])
        self.assertEqual(rows["dual"]["paired_speedup"]["median"],100/60)
        self.assertEqual(rows["dual"]["paired_speedup_vs_other_single"]["median"],2)
        self.assertIsNone(rows["single_other"]["paired_speedup_vs_other_single"])
        self.assertIsNone(rows["single"]["numeric_passed"])
        self.assertEqual(rows["single"]["numeric_role"],"reference")
        self.assertEqual(rows["dual"]["other_single_baseline_devices"],["cuda:0"])
        self.assertIn("另一单卡基线",render_html(report))

    def test_t5_queue_compares_both_hashes_and_does_not_reuse_unmatched_repeat(self):
        def queue(mode,repeat,wall,latent=True):
            row=dict(kind="dedicated_queue",mode=mode,frames=17,repeat=repeat,requests=2,wall_ms=wall,
                     videos_per_second=2000/wall,video_sha256=["v0","v1"],paired_exact=True)
            if latent:
                row["latent_sha256"]=["l0","l1"]
            return row
        self.write("encoder-v2/T5/result.json",dict(schema="ditserve.workflow.encoder.v1",variant="T5",status="completed",
            phase_records=[],requests=[queue("serial",0,100),queue("prefetch",0,80),queue("prefetch",1,1,False)]))
        report=build_report(self.root)
        row=next(row for row in report["metrics"]["dedicated_encoder_queues"] if row["mode"]=="prefetch")
        self.assertEqual(row["paired_speedup"]["median"],1.25)
        self.assertEqual(row["paired_speedup"]["n"],1)
        self.assertIsNone(row["numeric_passed"])
        self.assertIn("T5 专卡编码",render_html(report))

    def test_t5_queue_exactness_requires_latent_hashes_even_when_legacy_flag_true(self):
        rows=[dict(kind="dedicated_queue",mode=mode,frames=17,repeat=0,requests=1,wall_ms=100,
                   videos_per_second=10,video_sha256=["same"],paired_exact=True) for mode in ("serial","prefetch")]
        self.write("encoder-v2/T5/result.json",dict(schema="ditserve.workflow.encoder.v1",variant="T5",status="completed",requests=rows))
        row=next(row for row in build_report(self.root)["metrics"]["dedicated_encoder_queues"] if row["mode"]=="prefetch")
        self.assertIsNone(row["numeric_passed"])

    def test_explicit_include_excludes_unselected_files_and_html_escapes(self):
        path=self.write("encoder-v2/T0/result.json",self.encoder())
        other=self.encoder(12);other["variant"]="<script>alert(1)</script>"
        self.write("other/result.json",other)
        report=build_report(self.root,include=[path])
        self.assertEqual(len(report["metrics"]["sources"]),1)
        report["metrics"]["stage_resources"][0]["stage"]="<script>alert(1)</script>"
        report["metrics"]["stage_resources"][0]["allocator_device"]=None
        page=render_html(report)
        self.assertNotIn("<script>",page)
        self.assertIn("&lt;script&gt;",page)
        self.assertIn("未测 / 不适用",page)

    def test_missing_measurements_are_null(self):
        self.assertIsNone(distribution([None,float("nan")]))
        report=build_report(self.root)
        self.assertEqual(report["metrics"]["stage_resources"],[])
        self.assertTrue(all(row["status"]=="not_run" for row in report["phases"]))

    def test_superseded_before_execution_is_retained_without_poisoning_replacement(self):
        reason="Prior source retained VAE cache; replaced before CUDA execution."
        self.write("overlap/E5-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E5",
            status="superseded_before_execution",performance_evidence=False,reason=reason,planned_replacement="E5-f17-v4",
            config=dict(frames=17),evidence=dict(original_source="sources/v2"),
            pairs=[dict(modes={"streams":dict(wall_ms=999999)},correctness=False)]))
        self.write("overlap-v4/E5-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E5",status="complete",
            config=dict(frames=17),pairs=[dict(stage="vae",modes={"interleaved":dict(wall_ms=100),"streams":dict(wall_ms=80)},
                                            correctness=dict(streams=True))]))
        report=build_report(self.root)
        phase=next(row for row in report["phases"] if row["id"]=="E5")
        self.assertEqual(phase["status"],"completed")
        self.assertEqual(phase["metrics"]["source_count"],1)
        self.assertEqual(len(report["metrics"]["overlap_experiments"]),2)
        self.assertFalse(any(row["phase"]=="E5" for row in report["failures"]))
        marker=report["metrics"]["superseded_executions"][0]
        self.assertEqual(marker["reason"],reason)
        self.assertEqual(marker["planned_replacement"],"E5-f17-v4")
        self.assertFalse(marker["performance_evidence"])
        self.assertIn(reason,render_html(report))
        self.assertTrue(any(row["role"]=="superseded_before_execution" for row in report["metrics"]["sources"]))

    def test_reboot_supersession_retains_partial_raw_and_requires_full_replacement(self):
        old=self.overlap_job("E2",33)
        new=self.overlap_job("E2",33,"E2-f33-resume1",output="overlap-resume1/E2-f33")
        self.write("jobs.json",dict(jobs=[old]))
        self.write("jobs-resume.json",dict(jobs=[new]))
        raw=self.write("overlap/E2-f33/run.json",dict(self.overlap_result("E2",33),status="running"))
        original=raw.read_bytes()
        self.write("launches/E2-f33/launch.json",dict(id="E2-f33",status="interrupted",command=old["command"]))
        reason="Interrupted for user-requested reboot; partial raw outputs retained, full pair must restart"
        self.write("job-supersessions.json",dict(schema="ditserve.workflow.job_supersessions.v1",jobs=[
            dict(id="E2-f33",replacement="E2-f33-resume1",reason=reason)]))
        report=build_report(self.root)
        self.assertEqual(report["metrics"]["overlap_experiments"],[])
        self.assertEqual(report["metrics"]["planned_coverage"]["counts"]["normal"]["pending"],1)
        self.assertEqual(report["metrics"]["planned_coverage"]["counts"]["normal"]["expected"],1)
        self.assertEqual(report["metrics"]["superseded_executions"][0]["status"],"running")
        self.assertTrue(any(row["role"]=="superseded_interrupted" for row in report["metrics"]["sources"]))
        self.assertIn(reason,render_html(report))
        self.write("overlap-resume1/E2-f33/run.json",self.overlap_result("E2",33))
        report=build_report(self.root)
        self.assertEqual(next(row for row in report["phases"] if row["id"]=="E2")["status"],"completed")
        self.assertEqual(report["metrics"]["planned_coverage"]["counts"]["normal"]["completed"],1)
        self.assertEqual(raw.read_bytes(),original)

    def test_ncu_samples_are_separate_from_stage_and_performance_measurements(self):
        diagnostic=dict(schema="ditserve.workflow.ncu.v1",status="completed",coverage=dict(selected_kernel_count_fraction=None),
                        summaries=[dict(stage="dit_step",variant="T0",device="1",scope="sampled_kernels",profiled_kernel_count=2,
                                        summed_profiled_kernel_duration_ns=1000,metrics={"achieved_occupancy_percent":{"duration_weighted_mean":25}})])
        self.write("ncu/dit.json",diagnostic)
        self.write("probes/ncu-probe.json",dict(diagnostic,scope="probe"))
        report=build_report(self.root)
        self.assertEqual(len(report["metrics"]["kernel_diagnostics"]),1)
        self.assertEqual(report["metrics"]["stage_resources"],[])
        self.assertEqual(report["metrics"]["stage_hardware_counters"],[])
        self.assertIn("25",render_html(report))

    def test_ncu_capture_prefix_limit_frames_and_step_are_retained(self):
        capture=self.root/"ncu/dit.ncu-rep"
        capture.parent.mkdir(parents=True);capture.write_bytes(b"retained capture")
        stamp=capture.stat().st_mtime_ns
        self.write("launches/ncu-dit/launch.json",dict(status="completed",started_unix_ns=stamp-1,finished_unix_ns=stamp+1,
            command=["ncu","--launch-count","128","--range-filter","::1","--replay-mode","kernel","-o",str(capture.with_suffix(""))],
            gpu_monitor=dict(admission="passed",errors=[],conflicts=[],checks=2)))
        diagnostic=dict(schema="ditserve.workflow.ncu.v1",status="completed",source=dict(report=str(capture)),
            coverage=dict(selected_kernel_count_fraction=None,full_stage_time_fraction=None),
            kernels=[dict(stage="dit_step",device="1",included=True,metadata=dict(variant="T0",frames=17,step=1))],
            summaries=[dict(stage="dit_step",variant="T0",device="1",scope="sampled_kernels",profiled_kernel_count=128,
                            summed_profiled_kernel_duration_ns=None,metrics={})])
        self.write("ncu/dit.json",diagnostic)
        report=build_report(self.root)
        source=report["metrics"]["kernel_diagnostics"][0]
        self.assertEqual(source["selection"]["launch_count_limit"],128)
        self.assertEqual(source["selection"]["range_filter"],"::1")
        self.assertEqual(source["summaries"][0]["frames"],[17])
        self.assertEqual(source["summaries"][0]["dit_steps"],[1])
        self.assertIsNone(source["coverage"]["selected_kernel_count_fraction"])
        self.assertIsNone(source["summaries"][0]["summed_profiled_kernel_duration_ns"])
        rendered=render_html(report)
        self.assertIn("首 128 个 kernel 上限",rendered)
        self.assertIn("不代表全部文本编码",rendered)

    def test_e6_qualification_failure_completes_execution_without_performance_measurement(self):
        ranks=[dict(rank=rank,variants=[dict(mode="baseline",qualification=dict(steps=4,passed=False),runs=[])])
               for rank in (0,1)]
        self.write("overlap/E6-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E6",status="complete",
                                                config=dict(frames=17,steps=20),ranks=ranks))
        report=build_report(self.root)
        phase=next(row for row in report["phases"] if row["id"]=="E6")
        self.assertEqual(phase["status"],"completed")
        self.assertEqual(phase["qualification"]["scopes"]["numeric"]["status"],"failed")
        performance=phase["qualification"]["scopes"]["performance"]
        self.assertEqual(performance["status"],"unverified")
        self.assertEqual(performance["measured_scopes"],[])
        self.assertIn("not executed",performance["reason"])
        self.assertIsNone(report["metrics"]["overlap_experiments"][0]["wall_ms"])
        self.assertIn("不代表所有方案通过数值门槛",render_html(report))

    def test_e6_full_runs_and_encoder_only_timings_have_different_measured_scopes(self):
        ranks=[dict(rank=rank,variants=[dict(mode="baseline",qualification=dict(steps=4,passed=True),
            runs=[dict(repeat=0,warmup=False,wall_ms=100+rank,dit_ms=90+rank)])]) for rank in (0,1)]
        self.write("overlap/E6-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E6",status="complete",
                                                config=dict(frames=17,steps=20),ranks=ranks))
        self.write("encoder-v2/T1/result.json",dict(schema="ditserve.workflow.encoder.v1",variant="T1",status="completed",
            requests=[dict(kind="encoder",repeat=0,encode_ms=10)],numeric=[]))
        report=build_report(self.root)
        phases={row["id"]:row for row in report["phases"]}
        self.assertEqual(phases["E6"]["qualification"]["scopes"]["performance"]["status"],"measured")
        performance=phases["T1"]["qualification"]["scopes"]["performance"]
        self.assertEqual(performance["status"],"measured")
        self.assertEqual(performance["measured_scopes"],["encoder_only"])
        self.assertNotIn("full_video",performance["measured_scopes"])

    def launch(self, name, variant, status, started, finished, monitor=None, output="encoder-v2"):
        value=dict(id=name,status=status,started_unix_ns=started,finished_unix_ns=finished,
                   command=["python","-m","ditserve.evaluation.workflow_encoder","--output",f"/data/workflow/run01/{output}","--phases",variant])
        if monitor is not None:
            value["gpu_monitor"]=monitor
        return self.write(f"launches/{name}/launch.json",value)

    def test_completed_unmonitored_t0_is_not_invalidated_by_failed_t2(self):
        result=self.encoder();result.update(started_unix_ns=100,finished_unix_ns=150)
        self.write("encoder-v2/T0/result.json",result)
        self.launch("T0","T0","completed",90,160)
        self.launch("T2","T2","timeout",95,160,dict(conflicts=[]))
        report=build_report(self.root)
        self.assertTrue(report["metrics"]["stage_resources"])
        source=next(row for row in report["metrics"]["sources"] if row["role"]=="performance")
        self.assertEqual(source["validity"]["status"],"legacy_completed_unmonitored")

    def test_conflict_overrides_completed_child_and_is_excluded_from_resources(self):
        result=self.encoder();result.update(started_unix_ns=100,finished_unix_ns=150)
        self.write("encoder-v2/T0/result.json",result)
        self.launch("T0","T0","completed",90,160,dict(conflicts=[dict(processes=[dict(pid=999)])],errors=[],admission="passed"))
        report=build_report(self.root)
        self.assertEqual(report["metrics"]["stage_resources"],[])
        self.assertEqual(report["phases"][0]["status"],"invalidated")
        source=next(row for row in report["metrics"]["sources"] if row["role"]=="invalidated_performance")
        self.assertIn("foreign GPU work observed",source["validity"]["reasons"])

    def test_failed_earlier_and_later_attempts_do_not_poison_completed_retry(self):
        result=self.encoder();result.update(started_unix_ns=200,finished_unix_ns=220)
        self.write("encoder-v2/T0/result.json",result)
        old=self.launch("T0","T0","timeout",90,120)
        old.rename(old.with_name("attempt-1.json"))
        self.launch("T0","T0","completed",190,230,dict(conflicts=[],errors=[],admission="passed",checks=3))
        self.launch("T0-later","T0","gpu_monitor_unavailable",300,310,dict(errors=["lost monitor"]))
        report=build_report(self.root)
        source=next(row for row in report["metrics"]["sources"] if row["role"]=="performance")
        self.assertEqual(source["validity"]["status"],"monitored_valid")
        self.assertTrue(report["metrics"]["stage_resources"])

    def test_overlap_monotonic_clock_mapping_matches_timeout(self):
        self.write("overlap/E1-f17/run.json",dict(schema="ditserve.workflow-overlap.v1",experiment="E1",status="complete",
            started_ns=10,finished_ns=20,phases=[dict(start_ns=15,start_unix_ns=105)],config=dict(frames=17,steps=20),
            pairs=[dict(modes={"serial":dict(wall_ms=100),"prefetch":dict(wall_ms=80)},correctness=[dict(exact=True)])]))
        self.write("launches/E1/launch.json",dict(status="timeout",started_unix_ns=90,finished_unix_ns=120,
            command=["python","-m","ditserve.evaluation.workflow_overlap","--out=/data/workflow/run01/overlap/E1-f17"]))
        report=build_report(self.root)
        self.assertEqual(report["metrics"]["overlap_experiments"],[])
        invalid=next(row for row in report["metrics"]["sources"] if row["role"]=="invalidated_performance")
        self.assertEqual(invalid["validity"]["clock_basis"],"phase clock mapping")

    def test_embedding_replay_is_numerical_only_even_with_accidental_timing_field(self):
        self.write("numeric-replay/run.json",dict(schema="ditserve.workflow.embedding_replay.v1",status="completed",purpose="numerical_replay",
            include_in_performance=False,results=[dict(variant="T2",frames=17,steps=20,wall_ms=1,
                comparison=dict(video=dict(max_abs=.01,rmse=.001,cosine=.99,exact=False,finite=True,shape_match=True)))]))
        report=build_report(self.root)
        self.assertEqual(len(report["metrics"]["embedding_replay"]),1)
        self.assertEqual(report["metrics"]["stage_resources"],[])
        self.assertEqual(report["metrics"]["overlap_experiments"],[])
        self.assertIn("0.010",render_html(report))

    def test_stage_wall_share_pairs_request_timers_before_aggregation(self):
        value=self.encoder()
        value["requests"].append(dict(kind="video",request_id="T0-f17-r1",repeat=1,purpose="video_measurement",frames=17,inference_wall_ms=100))
        for row in self.encoder(30)["phase_records"][:2]:
            row["metadata"].update(request_id="T0-f17-r1",repeat=1)
            value["phase_records"].append(row)
        self.write("encoder-v2/T0/result.json",value)
        rows=build_report(self.root)["metrics"]["stage_resources"]
        row=next(row for row in rows if row["stage"]=="text_encoding")
        self.assertEqual(row["matched_denominator_requests"],2)
        self.assertEqual(row["request_wall_share_percent"]["median"],48.75)
        self.assertEqual(row["request_wall_denominator_ms"]["median"],58)

    def test_missing_request_timer_does_not_imply_zero_share(self):
        value=self.encoder();value["requests"]=[]
        self.write("encoder-v2/T0/result.json",value)
        row=build_report(self.root)["metrics"]["stage_resources"][0]
        self.assertIsNone(row["request_wall_share_percent"])
        self.assertIsNone(row["request_wall_denominator_ms"])
        self.assertEqual(row["matched_denominator_requests"],0)

    def test_memory_samples_match_device_and_disjoint_phase_windows(self):
        value=self.encoder()
        for index, phase in enumerate(value["phase_records"][:2]):
            phase.update(start_ns=100+200*index,end_ns=200+200*index,
                         torch_before=dict(device="cuda:1",allocated_bytes=100+index*100,device_used_bytes=450))
            phase["torch_after"]["device_used_bytes"]=600+index*300
        value["samples"]=[dict(timestamp_ns=t,device=d,memory_used_bytes=m,memory_total_bytes=2000)
                          for t,d,m in [(150,1,500),(170,0,9999),(250,1,9999),(350,1,700),(450,1,9999)]]
        self.write("encoder-v2/T0/result.json",value)
        report=build_report(self.root)
        row=next(row for row in report["metrics"]["stage_resources"] if row["stage"]=="text_encoding")
        self.assertEqual(row["stage_start_allocated_bytes"]["median"],100)
        self.assertEqual(row["peak_reserved_bytes"]["median"],1200)
        self.assertEqual(row["device_used_sampled_peak_bytes"]["median"],700)
        self.assertEqual(row["device_used_boundary_max_bytes"]["median"],900)
        self.assertEqual(row["device_used_sampled_capacity_fraction"]["median"],.35)
        self.assertEqual(row["nvml_samples_per_request"]["median"],2)
        self.assertEqual(row["nvml_observed_phase_windows"],2)
        self.assertIn("reserved 峰值 GiB",render_html(report))

    def test_missing_nvml_window_keeps_boundary_snapshot_separate(self):
        value=self.encoder()
        phase=value["phase_records"][0]
        phase.update(start_ns=100,end_ns=200,torch_before=dict(device="cuda:1",allocated_bytes=100,device_used_bytes=450))
        phase["torch_after"]["device_used_bytes"]=900
        value["samples"]=[dict(timestamp_ns=50,device=1,memory_used_bytes=9999,memory_total_bytes=20000)]
        self.write("encoder-v2/T0/result.json",value)
        row=next(row for row in build_report(self.root)["metrics"]["stage_resources"] if row["stage"]=="text_encoding")
        self.assertIsNone(row["device_used_sampled_peak_bytes"])
        self.assertIsNone(row["device_used_sampled_capacity_fraction"])
        self.assertEqual(row["device_used_boundary_max_bytes"]["median"],900)
        self.assertEqual(row["nvml_samples_per_request"]["median"],0)

    def test_recorded_gpu_uuid_takes_precedence_over_ordinal(self):
        value=self.encoder()
        value["cpu_affinity"]=dict(device="cuda:1",gpu_uuid="GPU-correct",cuda_visible_devices="1,0")
        value["phase_records"][0].update(start_ns=100,end_ns=200)
        value["samples"]=[dict(timestamp_ns=150,device=0,uuid="GPU-correct",memory_used_bytes=600),
                          dict(timestamp_ns=150,device=1,uuid="GPU-other",memory_used_bytes=9999)]
        self.write("encoder-v2/T0/result.json",value)
        row=next(row for row in build_report(self.root)["metrics"]["stage_resources"] if row["stage"]=="text_encoding")
        self.assertEqual(row["device_used_sampled_peak_bytes"]["median"],600)

    def test_execution_envelope_includes_async_gpu_tail_in_denominator(self):
        self.write("traces/async.json",dict(schema_version="ditserve.workflow_trace.v1",status="completed",
            availability=dict(cuda_kernels=True,cuda_memcpy=True),arms=[dict(name="async",host_span_ms=10,execution_span_ms=40,
                devices={"1":dict(compute_union_ms=32,communication={"d2h":dict(communication_ms=8)})})]))
        report=build_report(self.root)
        row=report["metrics"]["trace_overlap_summary"][0]
        self.assertEqual(row["envelope_ms"],40)
        self.assertEqual(row["host_envelope_ms"],10)
        self.assertEqual(row["compute_envelope_percent"],80)
        self.assertEqual(row["communication"]["d2h"]["envelope_percent"],20)
        self.assertIn("execution envelope",row["denominator_basis"])
        self.assertIn("执行包络",render_html(report))

    def test_trace_envelope_percentages_allow_actual_compute_communication_overlap(self):
        values=dict(compute_union_ms=80,cpu_encode_gpu_compute_overlap_ms=10,wan_sana_kernel_overlap_ms=20,
                    communication={"h2d":dict(communication_ms=60,overlap_ms=50,communication_coverage=5/6,bytes=1234),
                                   "nccl":dict(communication_ms=30,overlap_ms=20,communication_coverage=2/3,bytes=None)})
        self.write("traces/T0.json",dict(schema_version="ditserve.workflow_trace.v1",status="completed",
            availability=dict(cuda_kernels=True,cuda_memcpy=True),arms=[dict(name="T0/arm",host_span_ms=100,devices={"1":values})]))
        report=build_report(self.root)
        row=report["metrics"]["trace_overlap_summary"][0]
        self.assertEqual(row["compute_envelope_percent"],80)
        self.assertEqual(row["communication"]["h2d"]["envelope_percent"],60)
        self.assertEqual(row["communication"]["h2d"]["compute_overlap_ms"],50)
        self.assertEqual(row["status"],"available")
        self.assertIn("包络占比 %",render_html(report))

    def test_missing_cuda_tables_hide_placeholder_zero_overlap(self):
        values=dict(compute_union_ms=0,cpu_encode_gpu_compute_overlap_ms=0,wan_sana_kernel_overlap_ms=0,
                    communication={"h2d":dict(communication_ms=0,overlap_ms=0,communication_coverage=0,bytes=0),
                                   "nccl":dict(communication_ms=0,overlap_ms=0,communication_coverage=0,bytes=0)})
        self.write("traces/partial.json",dict(schema_version="ditserve.workflow_trace.v1",status="partial",
            availability=dict(cuda_kernels=False,cuda_memcpy=False),arms=[dict(name="missing",host_span_ms=100,devices={"1":values})]))
        row=build_report(self.root)["metrics"]["trace_overlap_summary"][0]
        self.assertEqual(row["status"],"unavailable")
        self.assertIsNone(row["compute_union_ms"])
        self.assertIsNone(row["compute_envelope_percent"])
        self.assertIsNone(row["cpu_encode_gpu_compute_overlap_ms"])
        for name in ("h2d","nccl"):
            self.assertIsNone(row["communication"][name]["union_ms"])
            self.assertIsNone(row["communication"][name]["communication_coverage"])
            self.assertIsNone(row["communication"][name]["bytes"])

    def test_memcpy_presence_without_kernels_reports_copy_but_not_overlap(self):
        self.write("traces/partial.json",dict(schema_version="ditserve.workflow_trace.v1",status="partial",
            availability=dict(cuda_kernels=False,cuda_memcpy=True),arms=[dict(name="copy-only",host_span_ms=10,
                devices={"1":dict(compute_union_ms=0,communication={"h2d":dict(communication_ms=2,overlap_ms=0,communication_coverage=0,bytes=12)})})]))
        row=build_report(self.root)["metrics"]["trace_overlap_summary"][0]
        self.assertEqual(row["status"],"partial")
        self.assertEqual(row["communication"]["h2d"]["union_ms"],2)
        self.assertEqual(row["communication"]["h2d"]["envelope_percent"],20)
        self.assertIsNone(row["communication"]["h2d"]["compute_overlap_ms"])
        self.assertIsNone(row["communication"]["h2d"]["communication_coverage"])

    def test_foreign_gpu_launch_invalidates_matching_profiler_capture(self):
        self.write("traces/T0-summary.json",dict(schema_version="ditserve.workflow_trace.v1",status="completed",
            source="/data/workflow/run01/traces/T0.sqlite",phases=[dict(name="text_encoder",included=True,
                metadata=dict(variant="T0",purpose="encoder_measurement"),devices={"1":dict(host_window_metrics={})})],arms=[]))
        self.write("launches/trace-T0/launch.json",dict(status="completed",started_unix_ns=100,finished_unix_ns=200,
            gpu_monitor=dict(conflicts=[dict(pid=999)],errors=[]),command=["/opt/nvidia/nsys","profile",
                "--output=/data/workflow/run01/traces/T0","python","-m","ditserve.evaluation.workflow_encoder",
                "--output","/data/workflow/run01/traces/encoder-T0","--phases","T0"]))
        report=build_report(self.root)
        self.assertEqual(report["metrics"]["trace_evidence"],[])
        self.assertEqual(report["metrics"]["stage_hardware_counters"],[])
        source=next(row for row in report["metrics"]["sources"] if row["role"]=="invalidated_trace_counters")
        self.assertIn("foreign GPU work observed",source["validity"]["reasons"])

    def test_failed_ncu_capture_is_excluded_but_unmonitored_diagnostic_is_retained(self):
        diagnostic=dict(schema="ditserve.workflow.ncu.v1",status="completed",summaries=[])
        self.write("ncu/dit.json",dict(diagnostic,source=dict(report="/data/workflow/run01/ncu/dit.ncu-rep")))
        self.write("ncu/other.json",dict(diagnostic,source=dict(report="/data/workflow/run01/ncu/other.ncu-rep")))
        self.write("launches/ncu/launch.json",dict(status="timeout",started_unix_ns=100,finished_unix_ns=200,
            command=["/opt/nvidia/ncu","-o","/data/workflow/run01/ncu/dit","python","-m","ditserve.evaluation.workflow_encoder",
                     "--output","/data/workflow/run01/ncu/encoder-T0","--phases","T0"]))
        report=build_report(self.root)
        self.assertEqual(len(report["metrics"]["kernel_diagnostics"]),1)
        invalid=next(row for row in report["metrics"]["sources"] if row["role"]=="invalidated_kernel_diagnostics")
        self.assertIn("launch status: timeout",invalid["validity"]["reasons"])
        valid=next(row for row in report["metrics"]["sources"] if row["role"]=="kernel_diagnostics")
        self.assertEqual(valid["validity"]["status"],"unmonitored_no_matching_ledger")

    def test_profiler_retry_matches_original_capture_time_not_later_json_import(self):
        self.write("traces/T0-summary.json",dict(schema_version="ditserve.workflow_trace.v1",status="completed",
            source="/data/workflow/run01/traces/T0.f17.sqlite",phases=[],arms=[]))
        capture=self.root/"traces/T0.f17.nsys-rep"
        capture.write_bytes(b"captured during first valid launch")
        os.utime(capture,ns=(150,150))
        command=["nsys","profile","-o","/data/workflow/run01/traces/T0.f17","python","-m",
                 "ditserve.evaluation.workflow_encoder","--output","/data/workflow/run01/traces/encoder","--phases","T0"]
        self.write("launches/trace/attempt-1.json",dict(status="completed",started_unix_ns=100,finished_unix_ns=200,command=command,
            gpu_monitor=dict(conflicts=[],errors=[],admission="passed")))
        self.write("launches/trace/launch.json",dict(status="timeout",started_unix_ns=300,finished_unix_ns=400,command=command))
        report=build_report(self.root)
        self.assertEqual(len(report["metrics"]["trace_evidence"]),1)
        source=next(row for row in report["metrics"]["sources"] if row["role"]=="trace_counters")
        self.assertEqual(source["validity"]["status"],"monitored_valid")
        self.assertEqual(source["validity"]["capture_unix_ns"],150)


if __name__ == "__main__":
    unittest.main()
