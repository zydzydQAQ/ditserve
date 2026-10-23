"""CPU-only dependency, actual-depth, numerical and summary contracts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_remaining_report as audit
from ditserve.evaluation import workflow_remaining_study as study
from ditserve.evaluation import workflow_elastic_validation as elastic
from ditserve.evaluation.workflow_serving_study import digest
from test_workflow_remaining_online_report import fixture, fingerprint
from test_workflow_remaining_online import inputs
from test_workflow_image_cache_audit import FakeAdapter, receipt
from ditserve.models.sana.backend import SanaBackend


def cpu_fixture():
    base, _, _ = fixture()
    context, profile, _ = inputs()
    template = study._cpu_templates(context,profile,.05)[0]
    config = template["config"]
    source = base["phases"][0]["cells"][0]
    videos = [r for r in source["requests"] if r["kind"]=="video"]
    phases = []
    with patch("ditserve.models.sana.runtime.SanaAdapter",FakeAdapter):
        for phase in config["phases"]:
            cell=copy.deepcopy(source);depth=phase["text_prefetch_depth"]
            cell.update(text_prefetch_depth=depth,scheduler_options=dict(phase["options"],degrees=[1],pin_started=True))
            for index,warm in enumerate(cell["warmup"]):
                warm["spec"]=dict(phase["warmup"][index]["spec"],request_id=f"warmup-{index}")
                if index<2:
                    warm.pop("negative_prompt_cache",None)
                    for event in warm["encoding_events"]:event.pop("negative_prompt_cache",None)
                else:
                    backend=SanaBackend(dict(config["workers"][warm["worker"]]["config"],model_path="/fake"))
                    warm["prompt_cache"]=receipt(backend,f"warmup-{index}",warm["spec"]["prompt"])
                    backend.close()
            rows=[]
            for index,planned in enumerate(phase["workload"]):
                old=next(r for r in videos if r["class"]==planned["class"])
                row=copy.deepcopy(old);row.update(copy.deepcopy(planned))
                row["spec"]["request_id"]=row["request_id"]
                offset=1000000.+index*10000.-old["start_ms"]
                for key in ("start_ms","finish_ms"):row[key]=old[key]+offset
                row["admission_ms"]=row["arrival_ms"]
                for event in row["dispatch_timeline"]:
                    for key in ("start_ms","finish_ms","rpc_completed_ms","harvested_ms"):event[key]+=offset
                    event["quantum_feedback"]["observed_at_ms"]+=offset
                row["outputs"]=[dict(output=fingerprint(row["request_id"]),latent=fingerprint("latent"+row["request_id"]))]
                row.pop("negative_prompt_cache",None)
                for event in row["encoding_events"]:event.pop("negative_prompt_cache",None)
                if depth==0:
                    row["encoding_events"]=row["encoding_events"][-1:]
                    row["encoding_events"][0].pop("ticket",None)
                    for key in ("encoder_rpc","encode_submit_ms","encode_ready_ms"):row.pop(key,None)
                    for key in ("encode_start","encode_poll"):row["operation_counts"].pop(key,None)
                else:
                    arrival=row["arrival_ms"]
                    row.update(encode_submit_ms=arrival+1,encode_ready_ms=arrival+3)
                    row["encoder_rpc"]["start"].update(start_ms=arrival,finish_ms=arrival+1)
                    row["encoder_rpc"]["ready_poll"].update(start_ms=arrival+2,finish_ms=arrival+3)
                rows.append(row)
            cell.update(requests=rows,elapsed_ms=max(r["finish_ms"] for r in rows)+100)
            phases.append(dict(id=phase["id"],status="completed",cells=[cell]))
    result=dict(schema="ditserve.run.v1",status="completed",config=copy.deepcopy(config),phases=phases)
    for worker in result["config"]["workers"].values():worker["gpu_uuid"]=f"GPU-{worker['gpu']}"
    job=dict(id=template["id"],metadata=template["metadata"],configuration_sha256=digest(config),
             gpu_mapping=[dict(ordinal=i,uuid=f"GPU-{i}") for i in (0,1)],devices=[0,1],requires=template["requires"])
    return result,job,config


def job_for(role,config,**metadata):
    uuids=config.get("gpu_uuids",[config.get("gpu_uuid","GPU-0"),"GPU-1"])
    return dict(id="test-"+role,metadata=dict(role=role,**metadata),configuration_sha256=digest(config),
                gpu_mapping=[dict(ordinal=i,uuid=u) for i,u in enumerate(uuids)],devices=[0],requires=["acceptance"])


class RemainingReportTests(unittest.TestCase):
    def test_pending_and_unknown_roles_fail_closed(self):
        for role in ("remaining_online","remaining_cpu_depth","remaining_interleave","remaining_sp_comparison",
                     "remaining_vae_lifecycle_bundle","remaining_vae_performance","remaining_vae_trace","unknown"):
            checked=audit.audit_result({},job_for(role,{}),{})
            self.assertFalse(checked["passed"],role)
            self.assertTrue(checked["errors"],role)

    def test_online_delegation_preserves_actual_receipts_and_work(self):
        result,job,config=fixture()
        checked=audit.audit_result(result,job,config)
        self.assertTrue(checked["passed"],checked["errors"])
        self.assertEqual(checked["arms"]["off300"]["work"]["observed_executed_blocks"],4800)

    def test_cpu_all_depths_synchronous_and_prefetch_are_distinct(self):
        result,job,config=cpu_fixture()
        checked=audit.audit_result(result,job,config)
        self.assertTrue(checked["passed"],checked["errors"])
        for arm in checked["arms"].values():
            self.assertEqual(arm["work"]["videos"],16)
            self.assertEqual(arm["work"]["observed_executed_blocks"],19200)
        metrics=audit._cpu_metrics(result,checked)
        self.assertTrue(metrics["safety"]["depth0/depth1"]["passed"])
        self.assertIsNone(checked["arms"]["depth0"]["timing"]["encoding_rpc_max_ms"])

    def test_cpu_mutations_are_rejected(self):
        result,job,config=cpu_fixture()
        for change in ("depth","synchronous_rpc","prefetch_ticket","blocks","hash","guard","cpu","frozen"):
            r,c,j=copy.deepcopy(result),copy.deepcopy(config),copy.deepcopy(job)
            cell=r["phases"][0]["cells"][0];row=cell["requests"][0]
            if change=="depth":cell["text_prefetch_depth"]=1
            elif change=="synchronous_rpc":row["operation_counts"]["encode_start"]=1
            elif change=="prefetch_ticket":r["phases"][1]["cells"][0]["requests"][0]["encoding_events"][0]["ticket"]="bad"
            elif change=="blocks":row["dispatch_timeline"][0]["executed_blocks"]-=1
            elif change=="hash":row["outputs"][0]["latent"]["sha256"]="f"*64
            elif change=="guard":cell["guard_telemetry"]["events"][-1]["owned_processes"]=[]
            elif change=="cpu":cell["worker_runtime"]["wan0"]["cpu_binding"]["torch_threads"]=16
            else:
                c["phases"][0]["options"]["quantum_ms"]=200;j["configuration_sha256"]=digest(c);r["config"]["phases"]=c["phases"]
            self.assertFalse(audit.audit_result(r,j,c)["passed"],change)

    def test_actual_add_job_native_envelope_is_accepted_and_bound(self):
        result,job,config=cpu_fixture()
        with tempfile.TemporaryDirectory() as directory:
            context=dict(root=directory,jobs=[],source="/frozen",source_digest="a"*64,gpu_mapping=job["gpu_mapping"])
            with patch.object(audit.api,"environment",return_value={}),patch.object(audit.api,"container_path",side_effect=lambda s,p:str(p)), \
                 patch.object(audit.api,"save"):
                registered=audit.api.add_job(context,job["id"],"remaining",config,native=True,
                    timeout=4200,requires=job["requires"],metadata=job["metadata"])
            frozen=json.loads(Path(registered["input"]).read_text())
            result["config"]=copy.deepcopy(frozen)
            for worker in result["config"]["workers"].values():worker["gpu_uuid"]=f"GPU-{worker['gpu']}"
            checked=audit.audit_result(result,registered,frozen)
            self.assertTrue(checked["passed"],checked["errors"])
            frozen["budget"]["watchdog_seconds"]=1;registered["configuration_sha256"]=digest(frozen)
            self.assertFalse(audit.audit_result(result,registered,frozen)["passed"])

    def test_interleave_requires_every_image_and_unchanged_wan_latent(self):
        config=elastic.normalize_config(dict(stage="interleave",seeds=[5401,5402],sp_linear_policy="full_rows_except_ffn1"))
        value=dict(sha256="a"*64,shape=[1,3,512,512],dtype="float32",finite=True)
        cases=[dict(sana_reference=value,gates=[dict(gate="sp2_sana_interleave",sana_interruptions=[
            dict(step=s,after_blocks=23,passed=True,wan_latent_unchanged=True,output=value,reference=value)
            for s in (0,5,10,15)])]) for _ in range(4)]
        result=dict(configuration=config,ranks=[dict(rank=0,cases=cases),dict(rank=1,cases=[dict(gates=[
            dict(gate="sp2_sana_interleave",sana_interruptions=[])]) for _ in range(4)])])
        job=job_for("remaining_interleave",config)
        with patch.object(elastic,"validate_repaired_result",return_value=dict(passed=True,errors=[])):
            self.assertTrue(audit.audit_result(result,job,config)["passed"])
            bad=copy.deepcopy(result);bad["ranks"][0]["cases"][0]["gates"][0]["sana_interruptions"][1]["wan_latent_unchanged"]=False
            self.assertFalse(audit.audit_result(bad,job,config)["passed"])
            bad=copy.deepcopy(result);bad["ranks"][1]["cases"][0]["gates"][0]["sana_interruptions"]=[{}]
            self.assertFalse(audit.audit_result(bad,job,config)["passed"])

    def test_sp_validator_and_scoped_timing_are_reused(self):
        from test_workflow_sp_comparison import fixture as sp_fixture
        result=sp_fixture();config=result["configuration"]
        job=job_for("remaining_sp_comparison",config,seed=5201)
        checked=audit.audit_result(result,job,config)
        self.assertTrue(checked["passed"],checked["errors"])
        bad=copy.deepcopy(result);bad["blocks"][0]["ranks"][0]["trials"][0]["passed"]=False
        self.assertFalse(audit.audit_result(bad,job,config)["passed"])

    def test_normal_dependencies_need_qualification_and_capacity(self):
        job=dict(id="target",requires=["dep"],metadata={})
        with patch.object(audit.report,"current_job",return_value=dict(id="dep")), \
             patch.object(audit.report,"evidence",return_value=dict(qualified=True,capacity=False)):
            checked=audit.prerequisite_gate({},job)
            self.assertFalse(checked["passed"]);self.assertEqual(checked["errors"],checked["reasons"])
        self.assertFalse(audit.prerequisite_gate({},dict(job,requires=["target"]))["passed"])
        self.assertFalse(audit.prerequisite_gate({},dict(job,requires=[]))["passed"])

    def test_cpu_dependencies_also_require_original_cross_depth_exactness(self):
        job=dict(id="cpu",requires=["dep"],metadata=dict(role="remaining_cpu_depth"))
        with patch.object(audit.report,"current_job",return_value=dict(id="dep")), \
             patch.object(audit.report,"evidence",return_value=dict(qualified=True,capacity=True)), \
             patch.object(audit.report,"screening_exact",return_value=False) as gate:
            self.assertFalse(audit.prerequisite_gate({},job)["passed"])
            gate.assert_called_once_with({},cross_cores=False)
            gate.return_value=True
            self.assertTrue(audit.prerequisite_gate({},job)["passed"])

    def test_bundle_selected_case_survives_other_case_failure_but_not_structure_or_guard(self):
        from test_workflow_overlap_bundle import BundleTests
        helper=BundleTests();self.addCleanup(helper.doCleanups)
        _,result,_,_=helper.run_mock(failed_case="a33-b17")
        config=result["configuration"];case=config["cases"][0]
        dependency=job_for("remaining_vae_lifecycle_bundle",config)
        dependency["gpu_mapping"]=[dict(ordinal=i,uuid=config["cases"][i]["gpu_uuid"]) for i in (0,1)]
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"bundle.json";source.write_text(json.dumps(config));dependency["input"]=str(source)
            target=Path(directory)/"performance.json";target.write_text(json.dumps(case["config"]))
            job=dict(id="target",input=str(target),requires=[dependency["id"]],devices=[case["device_ordinal"]],
                     metadata=dict(lifecycle_case=case["id"]))
            ev=dict(guarded=True,qualified=False,capacity=False,result=result,
                    remaining_prerequisites=dict(passed=True),remaining_audit=dict(observer_affinity_verified=True))
            with patch.object(audit.report,"current_job",return_value=dependency),patch.object(audit.report,"evidence",return_value=ev):
                checked=audit.prerequisite_gate({},job)
                self.assertTrue(checked["passed"],checked)
                ev["guarded"]=False
                self.assertFalse(audit.prerequisite_gate({},job)["passed"])
                ev["guarded"]=True
                ev["remaining_prerequisites"]["passed"]=False
                self.assertFalse(audit.prerequisite_gate({},job)["passed"])
                ev["remaining_prerequisites"]["passed"]=True
                ev["remaining_audit"]["observer_affinity_verified"]=False
                self.assertFalse(audit.prerequisite_gate({},job)["passed"])
                ev["remaining_audit"]["observer_affinity_verified"]=True
                job["metadata"]["lifecycle_case"]="a33-b17"
                self.assertFalse(audit.prerequisite_gate({},job)["passed"])

    def test_empty_summary_does_not_promote_or_hide_failed_bridge_screen(self):
        checked=audit.summarize(dict(jobs=[],root="/absent"))
        self.assertFalse(checked["cpu_depth"]["qualified"])
        self.assertFalse(checked["sp_comparison"]["qualified"])
        self.assertFalse(checked["default_change"])
        self.assertFalse(checked["dispatch_refresh_bridge"]["promotion"])
        self.assertEqual(len(checked["dispatch_refresh_bridge"]["per_seed"]),3)
        self.assertTrue(checked["unavailable"])

    def test_bridge_retains_original_screen_and_actual_confirmation_gates(self):
        context=dict(jobs=[],root="/absent",selections=dict(dispatch_refresh_screening=dict(passed=False,image_p95_ratio=1.14)))
        with patch.object(audit.report,"current_job",side_effect=lambda s,j:dict(id=j)), \
             patch.object(audit.report,"evidence",return_value=dict(launch=dict(status="completed"),qualified=True,capacity=True,guarded=True)):
            value=audit.summarize(context)["dispatch_refresh_bridge"]
        self.assertFalse(value["original_screen"]["passed"])
        self.assertTrue(all(row["qualified"] and row["capacity"] for row in value["per_seed"]))
        self.assertFalse(value["promotion"])


if __name__=="__main__":unittest.main()
