"""Temporary stdlib fixtures only; no model, GPU, or existing result reads."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

from ditserve.research.common import atomic_json, digest, file_digest
from ditserve.optimizations import next_audit as audit
from ditserve.optimizations import next_report
from ditserve.optimizations.online import paired_gate


def workbook(path, tables):
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    book = ET.Element("workbook", xmlns=ns)
    sheets = ET.SubElement(book, "sheets")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for number, (name, rows) in enumerate(tables.items(), 1):
            ET.SubElement(sheets, "sheet", name=name, sheetId=str(number))
            sheet = ET.Element("worksheet", xmlns=ns)
            data = ET.SubElement(sheet, "sheetData")
            for index, row in enumerate(rows, 1):
                node = ET.SubElement(data, "row", r=str(index))
                for column, value in enumerate(row, 1):
                    if value is None:
                        continue
                    letters, rest = "", column
                    while rest:
                        rest, rem = divmod(rest-1, 26)
                        letters = chr(65+rem)+letters
                    cell = ET.SubElement(node, "c", r=letters+str(index), t="inlineStr")
                    ET.SubElement(ET.SubElement(cell, "is"), "t").text = str(value)
            archive.writestr("xl/worksheets/sheet%d.xml" % number, ET.tostring(sheet))
        archive.writestr("xl/workbook.xml", ET.tostring(book))


def trace(seed, pattern, load):
    arrival = lambda i:float(i*10 if pattern=="ordinary" else i//4*40)/load
    requests = [dict(request_id="r%03d"%i, kind="video" if i in (0,12) else "image",
        arrival_ms=arrival(i), deadline_ms=arrival(i)+1000.,
        service_ms=50. if i in (0,12) else 5., seed=seed+i) for i in range(24)]
    return dict(seed=seed, pattern=pattern, load=load, requests=requests, trace_sha256=audit._identity(requests))


def measured(trace, order, arm):
    requests = copy.deepcopy(trace["requests"])
    factor = .8 if arm == "candidate" else 1.
    for r in requests:
        r.update(arrived=True, status="completed", start_ms=r["arrival_ms"]+1.,
                 finish_ms=r["arrival_ms"]+1.+r["service_ms"]*factor)
    elapsed = 1000.*factor
    return dict(trace_sha256=trace["trace_sha256"], seed=trace["seed"], pattern=trace["pattern"], load=trace["load"],
        order_index=order, arm=arm, elapsed_ms=elapsed, requests=requests, events=[],
        policy="block-4",image_batch_size=1,encode_device="cpu",encode_on_scheduler=False,
        status="completed", complete=True, metrics=audit._metrics(requests,elapsed),
        image_metrics=audit._metrics([r for r in requests if r["kind"]=="image"],elapsed),
        video_metrics=audit._metrics([r for r in requests if r["kind"]=="video"],elapsed))


class FinalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/"new/run01"
        self.ledger = self.root/"new/gpu-budget.json"
        self.config = dict(gpu_uuid="GPU-fixture", v100_uuid="GPU-v100", direction1=dict(prompts=["kettle","pendulum"]), seeds=[101,102,201,202])
        self.workloads = dict(diagnosis=[], highload={str(load):[trace(s,p,load) for s in (101,102,103)
            for p in ("ordinary","burst")] for load in (.5,.7,.9)})
        for mixed in (self.workloads["highload"]["0.5"][3],self.workloads["highload"]["0.5"][0]):
            image = dict(mixed, requests=[r for r in mixed["requests"] if r["kind"]=="image"])
            image["trace_sha256"] = audit._identity(image["requests"])
            self.workloads["diagnosis"].append(dict(parent_trace_sha256=mixed["trace_sha256"], mixed=mixed, image_only=image))
        self.auth = self.bound(self.root/"authorization.json", dict(approved_by_user=True,
            scope="serving_next_validation", budget_unlimited=True, authorized_gpu_seconds=None))
        parents = [self.bound(self.root/("parent%d.json"%i), dict(active=None, used_gpu_seconds=float(i))) for i in range(3)]
        mathematics = {"python/fixture.py":hashlib.sha256(b"# temporary fixture only\n").hexdigest()}
        self.study = dict(config_sha256=digest(self.config), workloads_sha256=digest(self.workloads), authorization=self.auth,
            historical_ledgers=parents, online_baseline_binding=self.bound(self.root/"old-report.json", dict(online_qualified=True,
                archive_verified=True,matrix_complete=True,online_evaluation_complete=True)),
            cache_local_binding=dict(mathematical_source_files=mathematics,mathematical_source_sha256=digest(mathematics),
                bindings=dict(qualification=self.bound(self.root/"old-cache.json",dict(qualified=True)))))
        self.budget = dict(active=None, budget_unlimited=True, limit_gpu_seconds=None, config_sha256=digest(self.config),
            authorization_binding=self.auth, parents={p["path"]:p for p in parents}, sessions=[], used_gpu_seconds=0.,
            stage_used_gpu_seconds={p:0. for p in audit.PHASES})
        atomic_json(self.output/"study.json", self.study)
        atomic_json(self.output/"protocol.json",dict(config=self.config,sha256=digest(self.config)))
        atomic_json(self.output/"workloads.json", self.workloads)
        for phase in audit.PHASES:
            self.install(phase)
        atomic_json(self.ledger, self.budget)
        self.report = dict(output=str(self.output), errors=[], phases=[dict(phase=p,status="completed",errors=[]) for p in audit.PHASES],
            source_bindings=[], budget=dict(settled_gpu_seconds=self.budget["used_gpu_seconds"]))
        workbook(self.output/"results.xlsx", next_report._tables(self.report))
        excel = dict(path=str(self.output/"results.xlsx"),sha256=file_digest(self.output/"results.xlsx"),status="verified")
        atomic_json(self.output/"report.json", dict(self.report,excel=excel))
        atomic_json(self.output/"excel-verification.json",excel)
        self.collect = patch.object(next_report,"collect",side_effect=lambda *unused:copy.deepcopy(self.report))
        self.collect.start()
        self.addCleanup(self.collect.stop)

    def bound(self,path,data):
        atomic_json(path,data)
        return dict(path=str(path),sha256=file_digest(path))

    def install(self,phase):
        attempt = self.output/phase/"attempt00"
        if phase=="diagnosis" or phase.startswith("cache_load_"):
            expected = audit._expected(self.workloads,phase)
            rows = [measured(t,o,a) for (unused,o,a),t in expected.items()]
            data = dict(status="completed",rows=rows,evaluation_complete=True,online_qualified=False)
            if phase!="diagnosis":
                data["comparison"] = paired_gate(rows, expected_trace_ids=sorted({key[0] for key in expected}))
                data["online_qualified"] = data["comparison"]["passed"]
        elif phase.startswith("sana_") and phase!="sana_numerics":
            data = dict(status="stopped",candidate="vae_compile" if phase=="sana_vae" else "transformer_graph",
                rows=[],local_qualified=False,online_qualified=False,reason="untriggered fixture")
            if phase=="sana_graph":
                data["trigger_evidence"] = dict(triggered=False)
            else:
                data["rows"] = [dict(kind="measurement",order_index=0,arm="eager",input_id="p0-s101",repeat=0,
                    repeat_exact=True,output_sha256="a"*64,
                    service=dict(candidate="eager",finite=True,ms=4.,invocation_wall_ms=5.),
                    quality=dict(passed=False,metrics=dict(status="ok",finite=True,range_valid=True,frame_count=1,
                        lpips_status="ok",lpips_per_frame=[.02],lpips_mean=.02,lpips_p95=.02)))]
        else:
            data = dict(status="stopped",rows=[],online_qualified=False,reason="unverified fixture")
        atomic_json(attempt/"config.json",self.config)
        atomic_json(attempt/"study.json",self.study)
        evidence = attempt/audit.FILES.get(phase,phase+".json")
        atomic_json(evidence,data)
        source = self.output/"sources"/(phase+"-attempt00")
        file = source/"python/fixture.py"
        file.parent.mkdir(parents=True)
        file.write_text("# temporary fixture only\n")
        hashes = {"python/fixture.py":file_digest(file)}
        atomic_json(source/"manifest.json",dict(source_files=hashes,source_sha256=digest(hashes),
            config_sha256=digest(self.config),execution_root=str(source)))
        identity = dict(phase=phase,output=str(attempt),run_root=str(self.output),gpu_uuids=[self.config["gpu_uuid"]],
            config_sha256=digest(self.config),source_sha256=digest(hashes),workloads_sha256=digest(self.workloads),authorization_source=self.auth)
        start = float(len(self.budget["sessions"])*10)
        charge = dict(stage=phase,session_id=phase,identity=identity,gpu_count=1,status="completed",
            started_monotonic=start,ended_monotonic=start+10.,elapsed_seconds=10.,gpu_seconds=10.)
        self.budget["sessions"].append(charge)
        self.budget["used_gpu_seconds"]+=10.
        self.budget["stage_used_gpu_seconds"][phase]+=10.
        terminal = dict(status=data["status"],phase=phase,output=str(attempt),evidence_path=str(evidence),charge=charge,
            returncode=0,final_occupancy=dict(status="exclusive_observed",owned=[],foreign=[]),
            config_sha256=digest(self.config),source_sha256=digest(hashes),online_qualified=data["online_qualified"])
        atomic_json(attempt/"supervisor.json",terminal)
        atomic_json(self.output/phase/"terminal.json",terminal)

    def change(self,path,change):
        data=json.loads(path.read_text())
        change(data)
        atomic_json(path,data)

    def run_audit(self):
        return audit.final_audit(self.output,self.ledger)

    def test_complete_integrity_while_unverified_scientific_phases_stay_unqualified(self):
        result=self.run_audit()
        self.assertEqual(result["status"],"passed",result.get("error"))
        self.assertTrue(result["all_phases_terminal"])
        self.assertEqual(result["settled_gpu_seconds"],80.)
        self.assertEqual(result["phases"][0]["matrix"]["recorded_requests"],184)
        for phase in result["phases"]:
            if phase["phase"].startswith("cache_load"):
                self.assertEqual((phase["matrix"]["recorded_replays"],phase["matrix"]["recorded_requests"]),(24,576))
                self.assertEqual(len(phase["matrix"]["traces"]),6)
                self.assertEqual(len(phase["matrix"]["groups"]),16)
        self.assertFalse(result["phases"][1]["online_qualified"])
        self.assertTrue(result["dependency_closure_required"])
        self.assertFalse(result["raw_arrays_read"])
        self.assertFalse(result["deletion_authorized"])

    def test_active_or_missing_terminal_is_pending_before_evidence_reads(self):
        self.change(self.ledger,lambda d:d.update(active={"pid":1}))
        result=self.run_audit()
        self.assertEqual(result["status"],"pending")
        self.assertEqual(len(result["source_bindings"]),1)
        self.change(self.ledger,lambda d:d.update(active=None))
        (self.output/"conditional/terminal.json").unlink()
        self.assertEqual(self.run_audit()["status"],"pending")

    def test_parent_change_blocks_integrity(self):
        self.change(Path(self.study["historical_ledgers"][0]["path"]),lambda d:d.update(used_gpu_seconds=99))
        self.assertEqual(self.run_audit()["status"],"failed")

    def test_source_config_workload_or_full_lifetime_change_rejected(self):
        paths=[self.output/"sources/diagnosis-attempt00/python/fixture.py",
               self.output/"diagnosis/attempt00/config.json",self.output/"workloads.json",self.ledger]
        for index,path in enumerate(paths):
            old=path.read_bytes()
            if index==0:
                path.write_text("changed\n")
            elif index==1:
                self.change(path,lambda d:d.update(seeds=[1,2,3,4]))
            elif index==2:
                self.change(path,lambda d:d["diagnosis"][0]["mixed"]["requests"][0].update(seed=0))
            else:
                self.change(path,lambda d:d["sessions"][0].update(ended_monotonic=11.))
            with self.subTest(index=index):
                self.assertEqual(self.run_audit()["status"],"failed")
            path.write_bytes(old)

    def test_cached_request_metric_and_deadline_tampering_fail_independently_of_report(self):
        path=self.output/"cache_load_05/attempt00/cache_load_05.json"
        old=path.read_bytes()
        for change in (lambda d:d["rows"][0]["metrics"].update(on_time=99),
                       lambda d:d["rows"][0]["requests"][1].update(deadline_ms=99999),
                       lambda d:d["rows"].pop(),
                       lambda d:d["comparison"].update(passed=False)):
            self.change(path,change)
            self.assertEqual(self.run_audit()["status"],"failed")
            path.write_bytes(old)

    def test_numerics_cannot_promote_default(self):
        self.change(self.output/"sana_numerics/attempt00/sana-numerics.json",lambda d:d.update(performance_claims=True))
        self.assertEqual(self.run_audit()["status"],"failed")

    def test_excel_literal_cell_tampering_detected_even_with_rebound_hash(self):
        tables=next_report._tables(self.report)
        tables["Phases"][1][0]="wrong phase"
        workbook(self.output/"results.xlsx",tables)
        sha=file_digest(self.output/"results.xlsx")
        self.change(self.output/"report.json",lambda d:d["excel"].update(sha256=sha))
        self.change(self.output/"excel-verification.json",lambda d:d.update(sha256=sha))
        result=self.run_audit()
        self.assertEqual(result["status"],"failed")
        self.assertIn("Excel cell differs",result["error"])

    def test_report_staleness_and_missing_source_fail(self):
        path=self.output/"report.json"
        old=path.read_bytes()
        self.change(path,lambda d:d["budget"].update(settled_gpu_seconds=79.))
        self.assertEqual(self.run_audit()["status"],"failed")
        path.write_bytes(old)
        (self.output/"diagnosis/attempt00/diagnosis.json").unlink()
        self.assertEqual(self.run_audit()["status"],"failed")

    def test_interrupted_requests_remain_in_arrived_denominator(self):
        expected=audit._expected(self.workloads,"cache_load_05")
        key, source=next(iter(expected.items()))
        row=measured(source,key[1],key[2])
        row.update(status="interrupted",complete=False)
        row["requests"][2].update(status="failed",finish_ms=None)
        row["metrics"]=audit._metrics(row["requests"],row["elapsed_ms"])
        row["image_metrics"]=audit._metrics([r for r in row["requests"] if r["kind"]=="image"],row["elapsed_ms"])
        value=audit._matrix(dict(status="incomplete",rows=[row]),expected)
        self.assertFalse(value["matrix_complete"])
        self.assertEqual(value["groups"][0]["metrics"]["failed_or_incomplete"],1)
        self.assertEqual(value["groups"][0]["metrics"]["arrived"],24)

    def test_cleanup_foreign_process_blocks_archive(self):
        self.change(self.output/"diagnosis/attempt00/supervisor.json",lambda d:d["final_occupancy"].update(foreign=[{"pid":1}]))
        self.assertEqual(self.run_audit()["status"],"failed")

    def test_settled_failed_attempt_can_be_archived_without_qualification(self):
        self.change(self.ledger,lambda d:d["sessions"][0].update(status="failed"))
        for path in (self.output/"diagnosis/terminal.json",self.output/"diagnosis/attempt00/supervisor.json"):
            def fail(d):
                d.update(status="failed",returncode=1)
                d["charge"]["status"]="failed"
            self.change(path,fail)
        result=self.run_audit()
        self.assertEqual(result["status"],"passed",result.get("error"))
        self.assertFalse(result["phases"][0]["online_qualified"])

    def test_only_explicit_conditional_v100_is_two_device_weighted(self):
        def weighted(d):
            row=d["sessions"][-1]
            row.update(gpu_count=2,gpu_seconds=20.)
            row["identity"].update(auxiliary_encoder=True,gpu_uuids=["GPU-fixture","GPU-v100"])
            d.update(used_gpu_seconds=90.)
            d["stage_used_gpu_seconds"]["conditional"]=20.
        self.change(self.ledger,weighted)
        charge=json.loads(self.ledger.read_text())["sessions"][-1]
        for path in (self.output/"conditional/terminal.json",self.output/"conditional/attempt00/supervisor.json"):
            self.change(path,lambda d:d.update(charge=charge))
        self.report["budget"]["settled_gpu_seconds"]=90.
        workbook(self.output/"results.xlsx",next_report._tables(self.report))
        excel=dict(path=str(self.output/"results.xlsx"),sha256=file_digest(self.output/"results.xlsx"),status="verified")
        atomic_json(self.output/"report.json",dict(self.report,excel=excel))
        atomic_json(self.output/"excel-verification.json",excel)
        result=self.run_audit()
        self.assertEqual(result["status"],"passed",result.get("error"))
        self.assertEqual(result["settled_gpu_seconds"],90.)
        self.change(self.ledger,lambda d:d["sessions"][-1]["identity"].update(gpu_uuids=["GPU-fixture","GPU-other"]))
        self.assertEqual(self.run_audit()["status"],"failed")

    def local_data(self):
        candidate="vae_compile"
        rows=[]
        for order in (0,1):
            for arm in ("eager",candidate):
                for p in range(2):
                    for seed in (101,102,201,202):
                        for repeat in range(3):
                            item="p%d-s%d"%(p,seed)
                            row=dict(kind="measurement",order_index=order,arm=arm,input_id=item,repeat=repeat,
                                repeat_exact=True,output_sha256=digest([arm,item]),
                                service=dict(candidate=arm,finite=True,ms=80. if arm=="eager" else 60.,
                                             invocation_wall_ms=100. if arm=="eager" else 81.))
                            if repeat==0:
                                row["quality"]=dict(passed=True,metrics=dict(status="ok",finite=True,range_valid=True,
                                    frame_count=1,lpips_status="ok",lpips_per_frame=[0.],lpips_mean=0.,lpips_p95=0.))
                            rows.append(row)
        return dict(status="completed",candidate=candidate,rows=rows,round_speed_gains=[.19,.19],
                    quality_passed=True,performance_passed=True,local_qualified=False,online_qualified=False)

    def test_sana_96_32_complete_wall_and_quality_are_recomputed(self):
        data=self.local_data()
        result=audit._sana(data,self.config,audit.Reader(),self.output/"unused",self.workloads)
        self.assertEqual((result["local_rows"],result["quality_rows"]),(96,32))
        self.assertTrue(result["component_gate"])
        self.assertFalse(result["online_qualified"])
        for change in (lambda d:d["rows"].pop(),
                       lambda d:d.update(round_speed_gains=[.25,.25]),
                       lambda d:d["rows"][0]["quality"]["metrics"].update(range_valid=False),
                       lambda d:d["rows"][1].update(output_sha256="b"*64)):
            wrong=copy.deepcopy(data)
            change(wrong)
            with self.assertRaises(ValueError):
                audit._sana(wrong,self.config,audit.Reader(),self.output/"unused",self.workloads)

    def test_local_qualification_requires_bound_32step_candidate_recovery(self):
        data=self.local_data()
        data.update(local_qualified=True,capacity=dict(passed=True))
        with self.assertRaises(KeyError):
            audit._sana(data,self.config,audit.Reader(),self.output/"unused",self.workloads)

    def rejection_data(self, candidate="vae_compile", measured=False):
        phase="sana_vae" if candidate=="vae_compile" else "sana_graph"
        attempt=self.output/phase/"attempt00"
        source=self.output/"sources"/(phase+"-attempt00")
        frame=source/"python/ditserve/optimizations/next_sana.py"
        frame.parent.mkdir(parents=True,exist_ok=True)
        frame.write_text("# source bound temporary candidate fixture\n")
        classifier=source/"python/ditserve/optimizations/next_candidate_failures.py"
        classifier.write_text("# source bound temporary rejection fixture\n")
        manifest=json.loads((source/"manifest.json").read_text())
        manifest["source_files"].update({str(p.relative_to(source)):file_digest(p) for p in (frame,classifier)})
        manifest["source_sha256"]=digest(manifest["source_files"])
        atomic_json(source/"manifest.json",manifest)
        rows=[dict(kind="warmup",arm=arm,index=i,service=dict(candidate=arm,ms=5.,invocation_wall_ms=6.))
            for arm in (("eager",candidate) if measured else ("eager",)) for i in range(2)]
        if measured:
            for row in self.local_data()["rows"][:24]:
                row["split"]="holdout" if row["input_id"].endswith(("s201","s202")) else "development"
                rows.append(row)
            unit=dict(kind="measurement",order_index=0,arm=candidate,input_id="p0-s101",split="development",repeat=0)
        else:
            unit=dict(kind="warmup",arm=candidate,index=0)
        graph=candidate=="transformer_graph"
        kind="builtins.RuntimeError" if graph else "torch._dynamo.exc.Unsupported"
        message="CUDA error: operation not permitted when stream is capturing" if graph else "operator unsupported by fullgraph"
        error_type=kind.rsplit(".",1)[1]
        error_text=error_type+": "+message
        category="cuda_capture_unsupported" if graph else "compiler_unsupported"
        operation=dict(module="ditserve.optimizations.next_sana",owner="_TransformerGraph" if graph else "_CompiledRegion",
            operation="_capture" if graph else "__call__",source_path=str(frame),source_sha256=file_digest(frame))
        region=dict(name="decoder",fullgraph=True,dynamic=False,cudagraphs=False,original_module_preserved=True,
            wrapper_setup_ms=1.,graph_compile_count=0,records=[],execution_failures=[dict(error=error_text,attempted_call=1)])
        return dict(status="stopped",candidate=candidate,decision="未验证",active_unit=None,rows=rows,
            local_qualified=False,online_qualified=False,online_evaluation_complete=False,
            reason="candidate screening rejected: "+category,failure_reason=error_text,
            traceback='Traceback (most recent call last):\n  File "%s", line 1, in %s\n%s'%(frame,operation["operation"],error_text),
            mechanics=dict(candidate=candidate,status="failed",qualified=False,cleanup_failed=False,
                failures=[dict(candidate=candidate,error=error_text,vae_regions=[] if graph else [region])],
                transformer_graph=dict(compiled=False,capture_count=0,records=[dict(status="failed",cold_host_ms=2.)]) if graph else None),
            expected_failure=dict(classification=category,operation=operation,module=kind.rsplit(".",1)[0],error_type=error_type,
                message=message,exception_chain=[dict(type=kind,type_mro=[kind,"builtins.Exception","builtins.BaseException","builtins.object"],message=message)],
                failed_unit=unit,failed_invocation_wall_ms=12.,cleanup=dict(passed=True,hooks_restored=True,graph_state_released=True))),attempt,manifest

    def test_recognized_compile_failure_is_terminal_but_not_qualified(self):
        data,attempt,manifest=self.rejection_data()
        result=audit._sana(data,self.config,audit.Reader(),attempt,self.workloads)
        self.assertEqual((result["local_rows"],result["quality_rows"]),(0,0))
        self.assertEqual(result["expected_failure"]["classification"],"compiler_unsupported")
        self.assertFalse(result["component_gate"])
        self.assertFalse(result["local_qualified"])
        self.assertFalse(result["online_qualified"])
        atomic_json(attempt/"sana-candidate.json",data)
        self.change(self.ledger,lambda d:d["sessions"][1]["identity"].update(source_sha256=manifest["source_sha256"]))
        charge=json.loads(self.ledger.read_text())["sessions"][1]
        for path in (attempt/"supervisor.json",attempt.parent/"terminal.json"):
            self.change(path,lambda d:d.update(source_sha256=manifest["source_sha256"],charge=charge))
        self.assertEqual(self.run_audit()["status"],"passed")

    def test_candidate_rejection_retains_exact_partial_quality_denominator(self):
        data,attempt,unused=self.rejection_data(measured=True)
        result=audit._sana(data,self.config,audit.Reader(),attempt,self.workloads)
        self.assertEqual((result["local_rows"],result["quality_rows"]),(24,8))
        self.assertFalse(result["full_matrix"])
        self.assertEqual(result["recomputed_speed_gains"],[])
        self.assertFalse(result["component_gate"])
        for mutation in (lambda d:d["expected_failure"]["failed_unit"].update(repeat=1),
                         lambda d:d["rows"][2].update(arm="eager")):
            wrong=copy.deepcopy(data)
            mutation(wrong)
            with self.assertRaisesRegex(ValueError,"frozen"):
                audit._sana(wrong,self.config,audit.Reader(),attempt,self.workloads)

    def test_candidate_rejection_rejects_generic_or_fatal_wrapped_errors(self):
        data,attempt,unused=self.rejection_data()
        wrong=copy.deepcopy(data)
        wrong["expected_failure"].update(module="builtins",error_type="RuntimeError",message="generic compiler failure")
        wrong["expected_failure"]["exception_chain"]=[dict(type="builtins.RuntimeError",type_mro=["builtins.RuntimeError"],message="generic compiler failure")]
        with self.assertRaisesRegex(ValueError,"recognized"):
            audit._sana(wrong,self.config,audit.Reader(),attempt,self.workloads)
        for kind,message in (("ditserve.directions.budget.BudgetStop","budget interruption"),
                             ("builtins.ValueError","source mismatch"),("torch.OutOfMemoryError","allocation failed"),
                             ("builtins.RuntimeError","CUDA illegal memory access")):
            wrong=copy.deepcopy(data)
            wrong["expected_failure"]["exception_chain"].append(dict(type=kind,type_mro=[kind],message=message))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError,"fatal"):
                audit._sana(wrong,self.config,audit.Reader(),attempt,self.workloads)

    def test_candidate_rejection_cannot_hide_cleanup_failure_or_grant_qualification(self):
        data,attempt,unused=self.rejection_data()
        for mutation in (lambda d:d["expected_failure"]["cleanup"].update(passed=False),
                         lambda d:d["expected_failure"]["cleanup"].pop("hooks_restored"),
                         lambda d:d.update(quality_passed=True),lambda d:d.update(performance_passed=True),
                         lambda d:d.update(local_qualified=True),lambda d:d.update(online_qualified=True),
                         lambda d:d["mechanics"].update(cleanup_failed=True)):
            wrong=copy.deepcopy(data)
            mutation(wrong)
            with self.assertRaises(ValueError):
                audit._sana(wrong,self.config,audit.Reader(),attempt,self.workloads)

    def test_candidate_rejection_requires_bound_actual_operation_and_original_failure(self):
        data,attempt,unused=self.rejection_data()
        for mutation in (lambda d:d["expected_failure"]["operation"].update(source_sha256="0"*64),
                         lambda d:d["expected_failure"]["operation"].update(source_path="/tmp/forged.py"),
                         lambda d:d["expected_failure"]["operation"].update(owner="Encoder"),
                         lambda d:d.update(traceback="no original trace"),
                         lambda d:d["mechanics"].update(failures=[]),
                         lambda d:d["expected_failure"]["failed_unit"].update(arm="eager")):
            wrong=copy.deepcopy(data)
            mutation(wrong)
            with self.assertRaises(ValueError):
                audit._sana(wrong,self.config,audit.Reader(),attempt,self.workloads)

    def test_capture_failure_is_narrow_and_subclass_compile_lineage_is_recorded(self):
        data,attempt,unused=self.rejection_data("transformer_graph")
        result=audit._sana(data,self.config,audit.Reader(),attempt,self.workloads)
        self.assertEqual(result["expected_failure"]["classification"],"cuda_capture_unsupported")
        data["expected_failure"]["operation"]["operation"]="__call__"
        with self.assertRaisesRegex(ValueError,"operation"):
            audit._sana(data,self.config,audit.Reader(),attempt,self.workloads)
        data,attempt,unused=self.rejection_data()
        failure=data["expected_failure"]
        failure.update(module="torch._dynamo.exc",error_type="SpecialUnsupported")
        failure["exception_chain"][0]["type"]="torch._dynamo.exc.SpecialUnsupported"
        failure["exception_chain"][0]["type_mro"].insert(0,"torch._dynamo.exc.SpecialUnsupported")
        data["failure_reason"]="SpecialUnsupported: "+failure["message"]
        data["traceback"]=data["traceback"].replace("Unsupported: ","SpecialUnsupported: ")
        data["mechanics"]["failures"][-1]["error"]=data["failure_reason"]
        result=audit._sana(data,self.config,audit.Reader(),attempt,self.workloads)
        self.assertEqual(result["expected_failure"]["classification"],"compiler_unsupported")

    def test_unstructured_empty_sana_failure_stays_invalid(self):
        data,attempt,unused=self.rejection_data()
        data.pop("expected_failure")
        with self.assertRaisesRegex(ValueError,"explicit graph trigger"):
            audit._sana(data,self.config,audit.Reader(),attempt,self.workloads)

    def test_metrics_sheet_reordering_is_representation_only(self):
        tables=next_report._tables(self.report)
        tables["Metrics"][1:]=reversed(tables["Metrics"][1:])
        path=self.root/"reordered.xlsx"
        workbook(path,tables)
        result=audit._excel(path,next_report._tables(self.report),audit.Reader())
        self.assertEqual(result["status"],"verified")

    def test_fresh_import_does_not_load_frameworks(self):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(Path(audit.__file__).resolve().parents[2]))
        result=subprocess.run([sys.executable,"-c",
            "import sys; import ditserve.optimizations.next_audit; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_conditional_untriggered_components_need_no_models_or_combined_gate(self):
        session=copy.deepcopy(self.budget["sessions"][-1])
        attempt=Path(session["identity"]["output"])
        trigger=dict(vae_trigger=False,encoder_trigger=False,triggered=False,admitted_sources=[],immutable_bindings=[])
        admission=dict(devices=["GPU-fixture"],auxiliary_encoder=False,trigger_evidence=trigger)
        session["identity"]["conditional_admission"]=self.bound(attempt/"conditional-admission.json",admission)
        data=dict(status="completed",trigger_evidence=trigger,admitted_devices=["GPU-fixture"],
            local_qualified=False,online_qualified=False,independently_qualified=[],
            video_vae=dict(status="stopped",reason="trigger absent",local_qualified=False,online_qualified=False),
            text_encoder=dict(status="stopped",reason="trigger absent",local_qualified=False,online_qualified=False))
        result=audit._conditional(data,session,audit.Reader(),self.output,self.ledger,self.workloads,self.budget["sessions"])
        self.assertFalse(result["combined_qualified"])
        self.assertEqual(result["independently_qualified"],[])
        data["video_vae"]["online_qualified"]=True
        with self.assertRaises(ValueError):
            audit._conditional(data,session,audit.Reader(),self.output,self.ledger,self.workloads,self.budget["sessions"])

    def test_conditional_local_vae16_and_encoder8_are_independently_recomputed(self):
        quality=dict(passed=True,metrics=dict(status="ok",finite=True,range_valid=True,frame_count=81,
            lpips_status="ok",lpips_per_frame=[.01]*81,lpips_mean=.01,lpips_p95=.01))
        inputs=["p%d-s%d"%(p,s) for p in range(2) for s in (101,102)]
        vae=dict(rows=[dict(kind="measurement",input_id=i,round=o,arm=a,repeat=0,generation_completed=True,
            service=dict(ms=100. if a=="eager" else 80.),quality=copy.deepcopy(quality))
            for i in inputs for o in (0,1) for a in ("eager","compiled")],order_gains=[.2,.2])
        result=audit._stage_local(vae,"video_vae")
        self.assertTrue(result["passed"])
        self.assertEqual(result["samples"],16)
        vae["rows"][1]["quality"]["metrics"]["range_valid"]=False
        with self.assertRaises(ValueError):
            audit._stage_local(vae,"video_vae")
        encoder=dict(rows=[dict(input_id=i,arm=a,device="cpu" if a=="cpu" else "cuda:1",
            encode_ms=10. if a=="cpu" else 8.,generation_ms=100. if a=="cpu" else 80.,completed=True,
            completed_steps=32,frames=81,cache="paired_008",policy="block-4",wan_backend="eager")
            for i in inputs for a in ("cpu","gpu")],
            pairs=[dict(input_id=i,order=["cpu","gpu"] if i.endswith("s101") else ["gpu","cpu"],
                quality=copy.deepcopy(quality),encode_gain=.2,generation_gain=.2) for i in inputs],
            encode_gain=.2,generation_gain=.2,local_qualified=True,encoder_device="cuda:1",encoder_dtype="float32",
            payload_device="cpu",payload_dtype="float32")
        result=audit._stage_local(encoder,"text_encoder")
        self.assertTrue(result["passed"])
        self.assertEqual(result["samples"],8)
        encoder["payload_dtype"]="bfloat16"
        with self.assertRaises(ValueError):
            audit._stage_local(encoder,"text_encoder")

    def conditional_fixture(self):
        session=copy.deepcopy(self.budget["sessions"][-1])
        attempt=Path(session["identity"]["output"])
        trigger=dict(vae_trigger=True,encoder_trigger=False,triggered=True)
        session["identity"]["conditional_admission"]=self.bound(attempt/"conditional-admission.json",
            dict(devices=["GPU-fixture"],auxiliary_encoder=False,trigger_evidence=trigger))
        quality=dict(passed=True,metrics=dict(status="ok",finite=True,range_valid=True,frame_count=81,
            lpips_status="ok",lpips_per_frame=[.01]*81,lpips_mean=.01,lpips_p95=.01))
        inputs=["p%d-s%d"%(p,s) for p in range(2) for s in (101,102)]
        paths=dict(valid=True,full_reuse_observed=False,counts=dict(full=32,reuse=0,invalid=0),
            path_sequence=[dict(step_index=i,completed_steps=i+1,path="compute",reused_forwards=0,outcome="full") for i in range(32)])
        expected=audit._expected(self.workloads,"sana")
        rows=[measured(t,o,a) for (unused,o,a),t in expected.items()]
        online=dict(status="completed",evaluation_complete=True,rows=rows,
            comparison=paired_gate(rows,expected_trace_ids=sorted({k[0] for k in expected})))
        online_path=attempt/"video-vae-online.json"
        atomic_json(online_path,online)
        native=dict(status="passed",local_qualified=True,online_qualified=True,online_evaluation_complete=True,
            online_path=str(online_path),order_gains=[.2,.2],
            rows=[dict(kind="measurement",input_id=i,round=o,arm=a,repeat=0,generation_completed=True,
                service=dict(ms=100. if a=="eager" else 80.),quality=copy.deepcopy(quality))
                for i in inputs for o in (0,1) for a in ("eager","compiled")],
            paired_latent_sources=[dict(request_id="stage-"+i+"-vae-latent",seed=int(i.rsplit("s",1)[1]),
                completed_steps=32,quantum=4,policy="paired_008",
                cache_paths=copy.deepcopy(paths)) for i in inputs])
        producer=dict(status="passed",local_qualified=True,online_qualified=True,system_qualified=True,
            online_evaluation_complete=True,decoder_cleanup_passed=True,online_path=str(online_path),
            local_evidence=self.bound(attempt/"vae.json",native),
            compiler=dict(compilation=[dict(status="completed",graph_invocations=12)],graph_compile_count=1,
                graph_invocations=12,compiled_forward_calls=12,callable_counting=True))
        data=dict(status="completed",trigger_evidence=trigger,admitted_devices=["GPU-fixture"],
            local_qualified=False,online_qualified=False,independently_qualified=["video_vae"],
            text_encoder=dict(status="stopped",reason="no encoder trigger",local_qualified=False,online_qualified=False))
        self.bind_component(data,producer,attempt)
        return data,producer,session,attempt

    def bind_component(self,data,producer,attempt):
        data["video_vae"]=dict(producer,triggered=True,
            component_evidence=self.bound(attempt/"conditional-video-vae.json",producer))

    def check_conditional(self,data,session):
        with patch.object(audit,"_triggers",return_value=dict(vae_trigger=True,encoder_trigger=False)):
            return audit._conditional(data,session,audit.Reader(),self.output,self.ledger,self.workloads,self.budget["sessions"])

    def test_conditional_vae_qualified_but_not_combination_and_all_full_input_is_valid(self):
        data,producer,session,attempt=self.conditional_fixture()
        result=self.check_conditional(data,session)
        self.assertEqual(result["independently_qualified"],["video_vae"])
        self.assertFalse(result["combined_qualified"])
        component=result["components"]["video_vae"]
        self.assertEqual(component["online"]["recorded_requests"],192)
        self.assertEqual(component["online"]["recorded_replays"],8)
        producer["online_evaluation_complete"]=False
        self.bind_component(data,producer,attempt)
        with self.assertRaisesRegex(ValueError,"recorded complete"):
            self.check_conditional(data,session)

    def test_conditional_cleanup_withdrawal_keeps_measured_facts_without_promotion(self):
        data,producer,session,attempt=self.conditional_fixture()
        producer.update(status="failed",local_qualified=False,online_qualified=False,system_qualified=False,
                        decoder_cleanup_passed=False,cleanup_error="fixture restoration failure")
        data.update(status="incomplete",independently_qualified=[])
        self.bind_component(data,producer,attempt)
        result=self.check_conditional(data,session)
        item=result["components"]["video_vae"]
        self.assertTrue(item["qualification_withdrawn"])
        self.assertTrue(item["measured_local_qualified"])
        self.assertTrue(item["online"]["online_qualified"])
        self.assertFalse(item["online_qualified"])
        self.assertEqual(result["independently_qualified"],[])
        producer.update(local_qualified=True,online_qualified=True,system_qualified=True)
        self.bind_component(data,producer,attempt)
        with self.assertRaisesRegex(ValueError,"cleanup failure"):
            self.check_conditional(data,session)

    def test_interrupted_partial_has_independent_root_source_pointer(self):
        expected=audit._expected(self.workloads,"cache_load_05")
        key,trace=next(iter(expected.items()))
        row=measured(trace,key[1],key[2])
        row.update(status="interrupted",complete=False)
        attempt=self.output/"cache_load_05/attempt00"
        path=attempt/"partial.json"
        self.bound(path,row)
        data=dict(status="incomplete",rows=[],active_unit=dict(partial_path=str(path)))
        matrix=audit._matrix(audit._with_partial(data,audit.Reader(),attempt),expected)
        self.assertEqual(matrix["canonical_rows"][0]["pointer"],"")
        self.assertEqual(matrix["canonical_rows"][0]["source"]["path"],str(path))
        self.assertEqual(matrix["canonical_rows"][0]["sha256"],digest(row))

    def test_conditional_trigger_recomputes_attribution_and_rejects_mutable_ledger_binding(self):
        from ditserve.optimizations.online import attribution
        phase="diagnosis"
        path=self.output/phase/"attempt00/diagnosis.json"
        native=json.loads(path.read_text())
        for row in native["rows"]:
            row["events"]=[dict(kind="scheduler_idle",reason="waiting_encoding",start_ms=0.,end_ms=100.)]
            row["attribution"]=attribution(row["requests"],row["events"],row["elapsed_ms"])
        source=self.bound(path,native)
        terminal_path=self.output/phase/"terminal.json"
        terminal=dict(path=str(terminal_path),sha256=file_digest(terminal_path))
        source_row=dict(phase=phase,source=source,terminal=terminal,session_id=phase,
            settled_session=self.budget["sessions"][0],vae_trigger=False,encoder_trigger=True,
            rows=[dict(pointer="/rows/%d"%i,**{key:row[key] for key in
                ("trace_sha256","arm","order_index","seed","pattern","attribution")},
                trigger_arm=row["arm"]=="mixed") for i,row in enumerate(native["rows"])])
        value=dict(vae_trigger=False,encoder_trigger=True,triggered=True,admitted_sources=[source_row],
                   immutable_bindings=[source,terminal])
        result=audit._triggers(value,audit.Reader(),self.output,self.ledger,self.budget["sessions"],self.workloads)
        self.assertTrue(result["encoder_trigger"])
        value["admitted_sources"][0]["rows"][0]["attribution"]=dict(encoding_trigger=False)
        with self.assertRaisesRegex(ValueError,"attribution/identity"):
            audit._triggers(value,audit.Reader(),self.output,self.ledger,self.budget["sessions"],self.workloads)
        value["immutable_bindings"]=[dict(path=str(self.ledger),sha256=file_digest(self.ledger))]
        with self.assertRaisesRegex(ValueError,"mutable whole ledger"):
            audit._triggers(value,audit.Reader(),self.output,self.ledger,self.budget["sessions"],self.workloads)


if __name__=="__main__":
    unittest.main()
