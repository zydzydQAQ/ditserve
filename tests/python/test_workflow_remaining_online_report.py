"""Real CPU cache receipts and mutation tests for all new online arm types."""

import copy
import hashlib
import math
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_remaining_online as study
from ditserve.evaluation import workflow_remaining_online_report as audit
from ditserve.evaluation.workflow_serving_study import digest
from ditserve.models.sana.backend import SanaBackend
from test_workflow_image_cache_audit import FakeAdapter, receipt
from test_workflow_encoding_qualification import row as wan_row
from test_workflow_remaining_online import inputs


def fingerprint(text):
    return dict(sha256=hashlib.sha256(text.encode()).hexdigest(),finite=True)


def fixture(family="quantum_repeated",repeat=0):
    template=next(j for j in study.templates(*inputs()) if j["metadata"]["family"]==family
                  and j["metadata"]["stage"]=="confirmation" and j["metadata"]["repeat"]==repeat)
    config=template["config"]
    observed=[]
    with patch("ditserve.models.sana.runtime.SanaAdapter",FakeAdapter):
        for phase_index,phase in enumerate(config["phases"]):
            runtime,backends,warmed,wan_states={},{},[],[{},{}]
            for name,settings in phase["worker_overrides"].items():
                runtime[name]=dict(pid=1000+phase_index*10+len(runtime),
                    backend="sana" if name.startswith("sana") else "wan",measured=True,
                    cpu_binding=dict(configured=True,requested_affinity=settings["cpu_affinity"],
                        effective_affinity=settings["cpu_affinity"],requested_threads=settings["cpu_threads"],
                        torch_threads=settings["cpu_threads"]))
                if name.startswith("sana"):
                    backends[name]=SanaBackend(dict(settings,model_path="/fake"))

            def attach_wan(target,slot,identifier,spec,prefetch):
                packet=wan_row("wan",dict(request_id=identifier,spec=spec),"enabled",wan_states[slot],
                    dict(model_path="/models/wan2.1-t2v-1.3b"))["cache"]
                packet.update(instance_id=f"{phase_index+1:02x}{slot+1:02x}"*8,cumulative_includes_warmup=True,
                              cpu_prompt_encode_ms=10.,encode_call_ms=9.,lookup_ms=.1,clone_ms=.2,store_ms=.3)
                target.update(cpu_prompt_encode_ms=10.,negative_prompt_cache=packet)
                event=dict(op="prepare",worker=f"wan{slot}",pid=runtime[f"wan{slot}"]["pid"],
                           cpu_prompt_encode_ms=10.,negative_prompt_cache=copy.deepcopy(packet))
                if prefetch:
                    event["ticket"]="ticket-"+identifier
                    target["encoding_events"]=[dict(copy.deepcopy(event),op="encode_poll"),event]
                else:target["encoding_events"]=[event]

            for index,item in enumerate(phase["warmup"]):
                name=item["worker"];spec=dict(item["spec"],request_id=f"warmup-{index}")
                warm=dict(worker=name,worker_pid=runtime[name]["pid"],state_id=f"warmup-{index}",
                          spec=spec,load_ms=1.,elapsed_ms=100.)
                if index>=2:warm["prompt_cache"]=receipt(backends[name],f"warmup-{index}",spec["prompt"])
                else:attach_wan(warm,index,f"warmup-{index}",spec,False)
                warmed.append(warm)
            rows=copy.deepcopy(phase["workload"])
            base=max(r["arrival_ms"] for r in rows)+100000.
            for index,row in enumerate(rows):
                slot=index%2;name=("sana" if row["kind"]=="image" else "wan")+str(slot)
                start=base+index*10000.
                row.update(status="completed",start_ms=start,finish_ms=start+5000.,admission_ms=row["arrival_ms"],
                           worker_pids={name:runtime[name]["pid"]},degrees=[1])
                row["spec"]["request_id"]=row["request_id"]
                final=dict(output=fingerprint(row["request_id"]))
                if row["kind"]=="image":
                    packet=receipt(backends[name],row["request_id"],row["spec"]["prompt"],row["spec"]["seed"])
                    final["prompt_cache"]=packet
                    row.update(model_ms=packet["service_ms"],operation_counts=dict(prepare=1,advance=1,finish=1,release=1),
                               dispatch_timeline=[dict(quantum="atomic",start_ms=start+1,finish_ms=start+49)])
                else:
                    cost=phase["profile"]["classes"][row["class"]];options=phase["options"]
                    timeline=[]
                    if phase["policy"]=="static_sp1":
                        timeline=[dict(quantum="step") for _ in range(row["spec"]["steps"])]
                    else:
                        quantum=max(1,min(math.floor(options["quantum_ms"]/(cost["step_compute_ms"]/cost["blocks_per_step"])),cost["blocks_per_step"]))
                        blocks=20*cost["blocks_per_step"]
                        while blocks:
                            executed=min(quantum,blocks)
                            event=dict(quantum=quantum,executed_blocks=executed)
                            if options.get("decode_frames"):event.update(block_count=quantum,chosen_quantum_ms=options["quantum_ms"])
                            timeline.append(event);blocks-=executed
                    advances=len(timeline);remaining=cost["decode_chunks"]
                    while remaining:
                        frames=min(options["decode_frames"],remaining) if options["chunked_decode"] else remaining
                        event=dict(quantum="finish")
                        if options["chunked_decode"]:event["decode_frames"]=frames
                        timeline.append(event);remaining-=frames
                    for pos,event in enumerate(timeline):event.update(start_ms=start+pos,finish_ms=start+pos+1)
                    row.update(dispatch_timeline=timeline,operation_counts=dict(prepare=1,advance=advances,
                        finish=len(timeline)-advances,release=1,encode_start=1,encode_poll=2),model_ms=100.)
                    final["latent"]=fingerprint("latent"+row["request_id"])
                    attach_wan(row,slot,row["request_id"],row["spec"],True)
                    row["encoder_rpc"]=dict(calls=3,total_ms=3.,max_ms=1.,failed_calls=0,timing_scope="host_encoder_round_trip",
                        start=dict(start_ms=row["arrival_ms"],finish_ms=row["arrival_ms"]+1,failed=False),
                        ready_poll=dict(start_ms=row["arrival_ms"]+2,finish_ms=row["arrival_ms"]+3,failed=False))
                    row.update(encode_submit_ms=row["arrival_ms"]+1,encode_ready_ms=row["arrival_ms"]+3)
                for event in row["dispatch_timeline"]:
                    event.update(group=[slot],reason="fixed_fixture",timing_scope="host_dispatch_task",
                                 rpc_completed_ms=event["finish_ms"]-.25,harvested_ms=event["finish_ms"]-.125,harvest_delay_ms=.125)
                    if phase["options"].get("decode_frames"):
                        event["quantum_feedback"]=dict(observed_at_ms=event["start_ms"]-.1,waiting_images=0,
                            last_image_arrival_ms=None,oldest_image_wait_ms=0.,adaptive=False,mode="fixed",trigger="fixed",hot_window_ms=0.)
                row.update(quanta=len(row["dispatch_timeline"]),outputs=[final])
            guard_events=[dict(start_monotonic_seconds=t,finish_monotonic_seconds=t+.001,duration_ms=1.,
                status="exclusive_observed",occupancy_backend="nvml",owned_count=4,foreign_count=0,
                owned_processes=[dict(pid=r["pid"],pgid=r["pid"],gpu_uuid=f"GPU-{config['workers'][n]['gpu']}",memory_mib="1024.0")
                                 for n,r in runtime.items()]) for t in (994.,997.,1000.1)]
            guard=dict(schema="ditserve.guard-telemetry.v1",occupancy_backend="nvml",cadence_seconds=2.,
                probe_timeout_seconds=3.,fail_closed=True,scope="cell startup, warmup and execution; excludes shutdown",
                heartbeat_calls=100,lock_wait_total_ms=.5,lock_wait_max_ms=.1,heartbeat_total_ms=4.,heartbeat_max_ms=1.1,
                probe_calls=3,probe_total_ms=3.,probe_max_ms=1.,probe_errors=0,events=guard_events,dropped_events=0)
            cell=dict(status="completed",gpus=2,policy=phase["policy"],real_backends_pinned_sp1=True,real_sp=None,
                text_prefetch_depth=1,elapsed_ms=max(r["finish_ms"] for r in rows)+100.,requests=rows,warmup=warmed,
                worker_runtime=runtime,scheduler_options=dict(phase["options"],degrees=[1],pin_started=True),
                dispatch_refresh=False,host_origin_monotonic_seconds=1000.,guard_telemetry=guard)
            observed.append(dict(id=phase["id"],status="completed",cells=[cell]))
            for backend in backends.values():backend.close()
    result=dict(schema="ditserve.run.v1",status="completed",config=copy.deepcopy(config),phases=observed)
    for worker in result["config"]["workers"].values():worker["gpu_uuid"]=f"GPU-{worker['gpu']}"
    job=dict(id=template["id"],metadata=template["metadata"],configuration_sha256=digest(config),
             gpu_mapping=[dict(ordinal=i,uuid=f"GPU-{i}") for i in (0,1)],requires=template["requires"])
    return result,job,config


class RemainingOnlineReportTests(unittest.TestCase):
    def test_all_six_families_audit_actual_cpu_receipts_and_generic_work(self):
        for family in study.SEEDS:
            with self.subTest(family=family):
                result,job,config=fixture(family)
                checked=audit.audit_result(result,job,config)
                self.assertTrue(checked["passed"],checked["errors"])
                for name,arm in checked["arms"].items():
                    work=arm["work"]
                    self.assertEqual(work["requests"],40 if family in ("poisson","high_load") else 24)
                    if name=="S0":
                        self.assertEqual((work["observed_executed_blocks"],work["step_dispatches"],work["derived_step_blocks"]),(0,80,4800))
                    else:self.assertEqual(work["observed_executed_blocks"],9600 if work["requests"]==40 else 4800)
                if family=="decode":
                    self.assertEqual([checked["arms"][name]["work"]["decode_dispatches"] for name in ("decode1","decode2","decodeall")],[28,16,4])
                if family=="quantum_unique":
                    workers=checked["arms"]["on200"]["sana"].values()
                    self.assertEqual(sum(w["measured"]["misses"] for w in workers),20)
                    self.assertGreater(sum(w["counters"]["evictions"] for w in workers),0)

    def test_configuration_and_identity_drift_fails_despite_rehashed_input(self):
        result,job,config=fixture()
        for mutation in (lambda c:c["phases"][1]["options"].update(quantum_ms=100),
                         lambda c:c["phases"][0]["workload"][0].update(deadline_ms=1),
                         lambda c:c["phases"][2]["worker_overrides"]["sana0"]["options"].update(prompt_cache_sync_mode="fused")):
            c=copy.deepcopy(config);mutation(c)
            r=copy.deepcopy(result);r["config"]["phases"]=c["phases"]
            self.assertFalse(audit.audit_result(r,dict(job,configuration_sha256=digest(c)),c)["passed"])

    def test_work_counts_step_and_decode_cannot_be_forged(self):
        for family,arm,change in (("quantum_repeated","off200","blocks"),("scheduler","S0","step"),
                                  ("decode","decode2","decode"),("decode","decodeall","all")):
            result,job,config=fixture(family)
            cell=next(p["cells"][0] for p in result["phases"] if p["id"]==arm)
            row=next(r for r in cell["requests"] if r["kind"]=="video")
            if change=="blocks":row["dispatch_timeline"][0]["executed_blocks"]-=1
            elif change=="step":row["dispatch_timeline"].pop(0)
            else:
                event=next(e for e in row["dispatch_timeline"] if e["quantum"]=="finish")
                event["decode_frames"]=99 if change=="decode" else 1
            self.assertFalse(audit.audit_result(result,job,config)["passed"],change)

    def test_cache_owner_ticket_guard_cpu_hash_and_causality_mutations_fail(self):
        base,job,config=fixture()
        for change in ("cache","ticket","cpu","guard","hash","feedback","timing","warmup","missing"):
            result=copy.deepcopy(base);cell=result["phases"][-1]["cells"][0]
            video=next(r for r in cell["requests"] if r["kind"]=="video")
            image=next(r for r in cell["requests"] if r["kind"]=="image")
            if change=="cache":image["outputs"][0]["prompt_cache"]["sequence"]+=1
            elif change=="ticket":video["encoding_events"][1]["ticket"]="wrong"
            elif change=="cpu":cell["worker_runtime"]["wan0"]["cpu_binding"]["torch_threads"]=16
            elif change=="guard":cell["guard_telemetry"]["events"][-1]["owned_processes"][0]["pid"]+=1
            elif change=="hash":video["outputs"][0]["latent"]["sha256"]="f"*64
            elif change=="feedback":image["dispatch_timeline"][0]["quantum_feedback"]["observed_at_ms"]=1e10
            elif change=="timing":image["dispatch_timeline"][0]["rpc_completed_ms"]=1e10
            elif change=="warmup":cell["warmup"][2]["spec"]["prompt"]="wrong"
            else:result["phases"].pop()
            self.assertFalse(audit.audit_result(result,job,config)["passed"],change)
        self.assertFalse(audit.audit_result({},job,config)["passed"])

    def test_confirmation_safety_is_separate_from_benefit_and_screen(self):
        def rows(family="quantum_repeated"):
            values=[]
            for seed,gain in zip(study.SEEDS[family],(14.,12.,-2.)):
                values.append(dict(family=family,seed=seed,stage="confirmation",qualified=True,arms={},paired_changes={},interactions={},
                    performance_assessment={a+"/"+b:dict(safety_passed=True,image_p95_reduction_percent=gain,
                        mean_service_reduction_percent=0.) for a,b in study.pairs(family)}))
            return values
        r=rows();out=audit.family_assessment(r,"quantum_repeated")
        self.assertTrue(out["comparisons"]["off300/off200"]["promotable"])
        r[0]["performance_assessment"]["off300/off200"]["safety_passed"]=False
        out=audit.family_assessment(r,"quantum_repeated")["comparisons"]["off300/off200"]
        self.assertTrue(out["benefit_confirmed"]);self.assertFalse(out["promotable"])
        r.append(dict(family="quantum_repeated",seed=4600,stage="screening",qualified=False))
        self.assertEqual(audit.family_assessment(r,"quantum_repeated")["comparisons"]["off300/off200"],out)
        unique=audit.family_assessment(rows("quantum_unique"),"quantum_unique")["comparisons"]["off300/off200"]
        self.assertTrue(unique["safety_passed"]);self.assertFalse(unique["benefit_confirmed"])
        self.assertFalse(audit.family_assessment(r[:2],"quantum_repeated")["qualified"])

    def test_no_image_slo_loss_is_a_required_per_seed_safety_gate(self):
        result,job,config=fixture()
        checked=audit.audit_result(result,job,config)
        arms={name:audit._metrics(cell,checked["arms"][name]) for name,cell in audit.report.native_arms(result).items()}
        arms["off300"]["image"]["slo_attainment"]=.9
        arms["off200"]["image"]["slo_attainment"]=.8
        out=audit.performance_assessment(arms,"quantum_repeated")["off300/off200"]
        self.assertFalse(out["safety_passed"])
        self.assertFalse(out["safety"]["image_slo"])

    def test_pending_jobs_are_not_false_numeric_failures(self):
        jobs=[dict(id=j["id"],metadata=j["metadata"]) for j in study.templates(*inputs())]
        with patch.object(audit.report,"evidence",return_value=dict(launch={})),patch.object(audit,"audit_result",side_effect=AssertionError("pending")):
            out=audit.summarize(dict(jobs=jobs))
        self.assertEqual(len(out["per_seed"]),20)
        self.assertFalse(out["qualified"])
        self.assertTrue(all(r["audit"] is None for r in out["per_seed"]))

    def test_summary_rechecks_technical_dependencies_and_retains_paired_and_absolute_values(self):
        jobs=[];configs={};evidence={}
        for repeat in range(3):
            result,job,config=fixture(repeat=repeat)
            job["input"]=job["id"]
            configs[job["input"]]=config
            evidence[job["id"]]=dict(launch=dict(status="completed"),result=result,qualified=True,capacity=True)
            jobs.append(job)
        gate=dict(qualified=True,capacity=True)
        with patch.object(audit.report,"evidence",side_effect=lambda _,j:evidence.get(j["id"],gate)), \
             patch.object(audit.report,"current_job",side_effect=lambda _,name:dict(id=name)), \
             patch.object(audit,"read",side_effect=lambda name,_:configs[name]), \
             patch.object(audit.encoding,"raw_capacity",return_value=dict(passed=True)):
            result=audit.summarize(dict(jobs=jobs))
            family=result["families"]["quantum_repeated"]
            self.assertTrue(family["qualified"])
            self.assertFalse(result["all_confirmations_qualified"])
            self.assertIn("off300/off200",family["paired_changes_median"])
            self.assertIn("off300",family["absolute_arm_medians"])
            self.assertIn("maximum_latency_ms",family["paired_changes_median"]["off300/off200"]["image"])
            self.assertIn("cold_probe_ms",family["paired_changes_median"]["off300/off200"]["guard"])
            gate["capacity"]=False
            self.assertFalse(audit.summarize(dict(jobs=jobs))["families"]["quantum_repeated"]["qualified"])


if __name__=="__main__":unittest.main()
