"""CPU contracts for complete four-arm SP request timing and qualification."""

import copy
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_sp_comparison as workflow


def config(**changes):
    return workflow.normalize_config(dict(gpu_uuids=["GPU-00000000-0000-0000-0000-000000000001",
        "GPU-00000000-0000-0000-0000-000000000002"], cpu_masks={"cuda:0":list(range(8)),
        "cuda:1":list(range(60,68))}, seeds=[5201], **changes))


def digest(number):
    return dict(sha256=f"{number:064x}", shape=[1], dtype="float32", finite=True)


def replicas(group, value):
    return dict(passed=True, replicas=[dict(rank=rank, owner=rank in group,
                latent=value if rank in group else None) for rank in (0,1)])


def fixture():
    cfg = config()
    result = dict(schema=workflow.SCHEMA, role=workflow.ROLE, status="completed", configuration=cfg,
                  qualification=dict(passed=True), performance_evidence=False, blocks=[])
    ordinary = dict(passed=True, initial=digest(0), steps=[digest(i+1) for i in range(20)],
                    video=digest(100), timesteps=list(range(20)))
    for seed in cfg["seeds"]:
        for arm in workflow.arm_order(seed):
            block = dict(seed=seed, arm=arm, ranks=[])
            result["blocks"].append(block)
            for rank in (0,1):
                data = dict(rank=rank, seed=seed, arm=arm, status="completed", backend_closed=True,
                    ordinary_reference_before_distributed=True, process_lifecycle=dict(exitcode=0,reaped=True),
                    arithmetic=dict(torch_threads=8,interop_threads=1,matmul_tf32=False,cudnn_tf32=True,
                                    attention="flash",negative_prompt_cache=False),
                    cpu_affinity=dict(bound_cpus=cfg["cpu_masks"][cfg["devices"][rank]]),
                    cuda_visible_devices=",".join(cfg["gpu_uuids"]), references=[], trials=[],
                    backend_init=dict(native_unattached=True,sp_linear_policy="native",transport=None) if arm == "native_sp1"
                        else dict(initialized=True,elastic=True,sp_linear_policy=workflow.POLICY,transport="host_file_v1"))
                block["ranks"].append(data)
                for frames in cfg["frames"]:
                    case=f"seed{seed}-f{frames}";spec=workflow._spec(cfg,frames,seed,case)
                    data["references"].append(dict(seed=seed,frames=frames,case=case,spec=spec,
                        ordinary=dict(copy.deepcopy(ordinary),spec=spec),cross_card_sp1_exact=True))
                for frames, mode, scope, index in workflow.trial_matrix(cfg,seed):
                    key = f"seed{seed}-f{frames}-{arm}-{mode}-{scope}-{index}"
                    start, prepared, dit_end = 1000000000,1001000000,1040000000
                    wall = dict(native_sp1=60,repaired_sp1=65,fixed_sp2=50,resize5to10=55)[arm]+rank
                    end = start+wall*1000000
                    row = dict(state_id=key,arm=arm,mode=mode,scope=scope,index=index,seed=seed,frames=frames,
                        passed=True,backend_native=arm == "native_sp1",diagnostic_in_timing=mode == "diagnostic",
                        cpu_encoding_in_timing=scope == "full_request",timing_scope=workflow.TIMING_SCOPES[scope],
                        start_ns=start,prepared_ns=prepared,dit_end_ns=dit_end,end_ns=end,wall_ms=float(wall),
                        prepare_ms=1.,dit_and_transfer_ms=39.,steps=[],transitions=[],
                        release={name:True for name in ("states","continuations","ownership","tickets","runtime_states",
                                                      "scratch","geometry","pending")})
                    data["trials"].append(row)
                    if rank == 0:
                        row["prepare"] = dict(state_id=key,cpu_prompt_encode_ms=10.,gpu_prepare_ms=1.,model_ms=11.)
                        if scope == "preencoded":
                            row["preencoding"] = dict(start_ns=start-20000000,end_ns=start-10000000,receipt=dict(ready=True))
                        samples = [dict(timestamp_ns=t,uuid=uuid,memory_total_bytes=24*1024**3,
                                        memory_free_bytes=8*1024**3,memory_used_bytes=16*1024**3)
                                   for t in (start-1,end+1) for uuid in cfg["gpu_uuids"]]
                        row["memory"] = dict(samples=samples,errors=[])
                        row["capacity"] = workflow.capacity_gate(row["memory"],cfg,start,end)
                        row["after_release_memory"] = dict(samples=copy.deepcopy(samples[2:]),errors=[])
                        row["after_release_capacity"] = workflow.capacity_gate(row["after_release_memory"],cfg)
                        row["video"] = ordinary["video"]
                    for step in range(20):
                        group, epoch = workflow._layout(arm,step)
                        item = dict(step=step,group=group,epoch=epoch,owner=rank in group,
                                    start_ns=start+5000000+step*1000000,end_ns=start+5500000+step*1000000)
                        row["steps"].append(item)
                        if rank in group:
                            event = dict(kind="block_quantum",step_index=step,step_completed=True,blocks=60)
                            response = dict(events=[event],progress=dict(step=step+1))
                            if arm != "native_sp1":
                                response["progress"].update(group=group,ownership_epoch=epoch,quantum_seq=step+1)
                                event["linear_repair"] = dict(policy=workflow.POLICY,degree=len(group),request_id=key,
                                    step_index=step,quantum_seq=step,global_sequence=((frames-1)//4+1)*30*52,
                                    local_sequence=((frames-1)//4+1)*30*52//len(group),
                                    unchanged_sp1_calls=60 if len(group)==1 else 0,
                                    applied_calls=0 if len(group)==1 else 422)
                            item["response"] = response
                            if mode == "diagnostic":
                                item.update(exact=True,latent=ordinary["steps"][step])
                        if mode == "diagnostic":
                            item["replicas"] = replicas(group,ordinary["steps"][step])
                    if mode == "diagnostic" and (rank == 0 or arm == "fixed_sp2"):
                        row.update(initial=ordinary["initial"],timesteps=ordinary["timesteps"])
                    final_group,_ = workflow._layout(arm,19)
                    if rank in final_group:
                        row["final_latent"] = ordinary["steps"][-1]
                    row["final_replicas"] = replicas(final_group,ordinary["steps"][-1])
                    for n in range(1 if arm == "fixed_sp2" else 2 if arm == "resize5to10" else 0):
                        step = 0 if arm == "fixed_sp2" else (5 if n==0 else 10)
                        source,target = ([0],[0,1]) if n==0 else ([0,1],[0])
                        tx_start = prepared+100000 if step == 0 else row["steps"][step-1]["end_ns"]+100000
                        row["transitions"].append(dict(completed_steps=step,passed=True,start_ns=tx_start,end_ns=tx_start+100000,
                            checks={key:True for key in ("committed","identity","ownership","ready","digest","contract",
                                                       "state","continuation","record","progress")},
                            request=dict(state_id=key,source_group=source,target_group=target,expected_epoch=n,transfer_id=str(n)),
                            response=dict(phase="finalized",committed=True,state_id=key,transfer_id=str(n),group=target,
                                ownership_epoch=n+1,transfer_ready=True,state_sha256="a"*64,
                                execution_contract=dict(schema="ditserve.wan-sp-execution.v1",sp_linear_policy=workflow.POLICY),
                                progress=dict(step=step,group=target,ownership_epoch=n+1,transfer_ready=True) if rank in target else None)))
    return result


class ComparisonTests(unittest.TestCase):
    def test_configuration_binds_recipe_seed_subset_physical_pair_and_masks(self):
        self.assertEqual(config()["seeds"],[5201])
        cfg=config();cfg["seeds"]=[5202,5203]
        self.assertEqual(workflow.normalize_config(cfg)["seeds"],[5202,5203])
        changes = [dict(seeds=[]),dict(seeds=[5202,5201]),dict(seeds=[5201,5201]),dict(seeds=[3501]),
                   dict(frames=[17]),dict(warmup=1),dict(repeats=2),dict(exact_required=False),dict(arms=["fixed_sp2"]),
                   dict(gpu_uuids=[cfg["gpu_uuids"][0]]*2),dict(sp_linear_policy="native"),dict(timeout_seconds=None),
                   dict(cpu_masks={"cuda:0":list(range(8)),"cuda:1":list(range(8))}),dict(group_timeout_seconds=120)]
        for change in changes:
            bad=config();bad.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                workflow.normalize_config(bad)
        self.assertEqual([workflow.arm_order(s)[0] for s in workflow.SEEDS],list(workflow.ARMS[:3]))

    def test_complete_subset_and_both_scopes_are_auditable(self):
        result=fixture();audit=workflow.validate_result(result)
        self.assertTrue(audit["passed"],audit)
        result["summary"]=workflow.summarize(result)
        self.assertTrue(workflow.validate_result(result)["passed"])
        self.assertEqual(len(result["summary"]["pairs"]),12)
        self.assertFalse(result["summary"]["three_seed_complete"])
        arms=result["summary"]["pairs"][0]["arms"]
        self.assertEqual(arms["native_sp1"]["wall_ms"],61)
        self.assertEqual(arms["native_sp1"]["two_gpu_envelope_seconds"],.122)
        self.assertEqual(arms["native_sp1"]["assigned_group_gpu_seconds"],.06)
        self.assertGreater(arms["fixed_sp2"]["assigned_group_gpu_seconds"],.09)

    def test_durable_tampering_rejected(self):
        pristine=fixture()
        def trial(r,arm=0,rank=0,index=-1):return r["blocks"][arm]["ranks"][rank]["trials"][index]
        mutations=[lambda r:r["blocks"].pop(),lambda r:r["blocks"].reverse(),
            lambda r:r["blocks"][0]["ranks"].pop(),lambda r:trial(r).update(video=digest(999)),
            lambda r:trial(r).update(wall_ms=float("nan")),lambda r:trial(r).update(start_ns=1),
            lambda r:trial(r)["release"].update(tickets=False),lambda r:trial(r).update(diagnostic_in_timing=True),
            lambda r:trial(r)["steps"][2].update(latent=digest(3)),
            lambda r:trial(r,1)["steps"][5]["response"]["events"][0]["linear_repair"].update(policy="native"),
            lambda r:trial(r,3)["transitions"][0].update(start_ns=1),
            lambda r:trial(r,3)["transitions"][1]["response"].update(group=[0,1]),
            lambda r:trial(r,2)["final_replicas"]["replicas"][1].update(latent=digest(999)),
            lambda r:trial(r)["memory"]["samples"][0].update(uuid="wrong"),
            lambda r:trial(r)["memory"]["samples"][0].update(memory_free_bytes=100),
            lambda r:trial(r)["memory"].update(errors=["query failed"]),
            lambda r:trial(r,0,0,0)["steps"][1].update(exact=False),
            lambda r:trial(r,0,0,0)["preencoding"].update(end_ns=2000000000),
            lambda r:r["blocks"][0]["ranks"][0].update(cuda_visible_devices="wrong"),
            lambda r:r["blocks"][0]["ranks"][0]["arithmetic"].update(negative_prompt_cache=True),
            lambda r:r["blocks"][0]["ranks"][1]["process_lifecycle"].update(exitcode=9),
            lambda r:r.update(summary={})]
        for n,mutate in enumerate(mutations):
            result=copy.deepcopy(pristine);mutate(result)
            with self.subTest(mutation=n):self.assertFalse(workflow.validate_result(result)["passed"])

    def test_capacity_requires_both_cards_and_bracketed_request(self):
        row=fixture()["blocks"][0]["ranks"][0]["trials"][0]
        for change in (lambda m:m["samples"].pop(),lambda m:m["samples"][0].update(timestamp_ns=row["start_ns"]+1),
                       lambda m:m["samples"][0].update(memory_total_bytes=True)):
            memory=copy.deepcopy(row["memory"]);change(memory)
            self.assertFalse(workflow.capacity_gate(memory,config(),row["start_ns"],row["end_ns"])["passed"])

    def test_actual_trial_keeps_diagnostic_copies_outside_normal_timing_and_releases(self):
        import numpy as np
        from test_workflow_elastic_validation import CandidateTests
        for arm in workflow.ARMS:
            for mode,scope in (("diagnostic","preencoded"),("measured","preencoded"),("measured","full_request")):
                with self.subTest(arm=arm,mode=mode,scope=scope),tempfile.TemporaryDirectory() as temp:
                    backend,advances,transitions,released=CandidateTests().backend(policy=workflow.POLICY)
                    log=[];backend.tickets={};backend.runtime._states=backend.states
                    backend.runtime._sync=lambda:None
                    backend.runtime.torch.cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda device:None)
                    repair=types.SimpleNamespace(scratch=None,geometry=None,pending={})
                    backend.parallel=types.SimpleNamespace(handle=types.SimpleNamespace(rank=0),linear_policy=workflow.POLICY,
                                                          _linear_repair=repair)
                    if arm=="native_sp1":
                        backend.parallel=None;backend.elastic=None
                        advance=backend.advance
                        def native_advance(request):
                            response=advance(request);response["events"][0].pop("linear_repair");return response
                        backend.advance=native_advance
                    prepare=backend.prepare
                    def new_prepare(request):
                        log.append(("prepare",time.perf_counter_ns()))
                        if "encode_ticket" in request:backend.tickets.pop(request["encode_ticket"])
                        prepare(request)
                        return dict(state_id=request["state_id"],cpu_prompt_encode_ms=1.,gpu_prepare_ms=1.,model_ms=2.)
                    backend.prepare=new_prepare
                    def encode(request):
                        log.append(("encode",time.perf_counter_ns()))
                        backend.tickets["ticket"]=(request["spec"],types.SimpleNamespace(result=lambda:None))
                        return dict(ticket="ticket")
                    backend.encode_start=encode;backend.encode_poll=lambda request:dict(ready=True)
                    backend.encode_cancel=lambda request:backend.tickets.pop(request["ticket"],None)
                    backend.finish=lambda request:dict(output=dict(array_sha256=workflow._hash_array(np.array([20.],dtype=np.float32))["sha256"],
                        shape=[1],dtype="float32",finite=True))
                    root=Path(temp);reference=CandidateTests().reference(root)
                    ordinary=json.loads((reference/"reference.json").read_text())
                    ordinary.update(steps=[workflow._hash_array(np.array([i+1.],dtype=np.float32)) for i in range(20)],
                                    video=workflow._hash_array(np.array([20.],dtype=np.float32)))
                    (reference/"reference.json").write_text(json.dumps(ordinary))
                    sampler=types.SimpleNamespace(samples=[],errors=[])
                    def sample():
                        rows=[dict(timestamp_ns=time.perf_counter_ns(),uuid=uuid,memory_total_bytes=24*1024**3,
                            memory_free_bytes=8*1024**3,memory_used_bytes=16*1024**3) for uuid in config()["gpu_uuids"]]
                        sampler.samples.extend(rows)
                        return rows
                    sampler.sample_once=sample
                    copies=[];original=workflow._cpu_array
                    def cpu_array(tensor):copies.append(1);return original(tensor)
                    with patch.object(workflow,"_cpu_array",side_effect=cpu_array),patch.object(workflow,"torch_snapshot",return_value={}):
                        row=workflow._trial(backend,config(),0,arm,dict(request_id="test",frames=17,seed=5201),mode,scope,0,
                                            reference,root/"rank0",sampler)
                    self.assertTrue(row["passed"],row)
                    self.assertEqual(len(copies),22 if mode=="diagnostic" else 1)
                    self.assertEqual([a["quantum_seq"] for a in advances],list(range(20)))
                    self.assertEqual(len(transitions),1 if arm=="fixed_sp2" else 2 if arm=="resize5to10" else 0)
                    self.assertTrue(all(row["start_ns"]<=tx["start_ns"]<=tx["end_ns"]<=row["end_ns"] for tx in row["transitions"]))
                    self.assertTrue(all(row["release"].values()))
                    self.assertEqual(len(released),1)
                    self.assertLessEqual(row["start_ns"],next(t for name,t in log if name=="prepare"))
                    if scope=="preencoded":self.assertLess(next(t for name,t in log if name=="encode"),row["start_ns"])
                    else:self.assertFalse(any(name=="encode" for name,t in log))
                    if arm=="resize5to10" and mode=="measured" and scope=="full_request":
                        backend.finish=lambda request:dict(output=dict(array_sha256="0"*64,shape=[1],dtype="float32",finite=True))
                        with patch.object(workflow,"torch_snapshot",return_value={}),self.assertRaisesRegex(RuntimeError,"video is not exact"):
                            workflow._trial(backend,config(),0,arm,dict(request_id="test",frames=17,seed=5201),mode,scope,1,
                                            reference,root/"rank0",sampler)
                        failed=json.loads((root/"rank0"/"trials"/f"test-{arm}-{mode}-{scope}-1.json").read_text())
                        self.assertFalse(failed["passed"])
                        self.assertTrue(all(failed["release"].values()))

    def test_timeout_reaps_every_started_rank_and_late_failed_exit_is_rejected(self):
        class Process:
            def __init__(self,exitcode=None):
                self.exitcode=exitcode;self.started=False;self.terminated=False;self.joined=0
            def start(self):self.started=True
            def is_alive(self):return self.started and self.exitcode is None
            def join(self,timeout=0):self.joined+=1
            def terminate(self):self.terminated=True;self.exitcode=-15
            def kill(self):self.exitcode=-9
        children=[Process(),Process()]
        context=types.SimpleNamespace(Process=lambda **kw:children.pop(0))
        retained=list(children)
        with tempfile.TemporaryDirectory() as temp,patch.object(workflow.multiprocessing,"get_context",return_value=context),\
                patch.object(workflow.time,"monotonic",side_effect=[0.,0.,2.]):
            with self.assertRaises(TimeoutError):workflow._run_block(config(),5201,"native_sp1",Path(temp),1.)
        self.assertTrue(all(p.started and p.terminated and p.joined for p in retained))
        children=[Process(0),Process(7)]
        context=types.SimpleNamespace(Process=lambda **kw:children.pop(0))
        with tempfile.TemporaryDirectory() as temp,patch.object(workflow.multiprocessing,"get_context",return_value=context):
            root=Path(temp);(root/"rank0").mkdir();(root/"rank0"/"run.json").write_text(json.dumps(dict(status="completed")))
            with self.assertRaisesRegex(RuntimeError,"final exit 7"):
                workflow._run_block(config(),5201,"native_sp1",root,time.monotonic()+30)

    def test_failure_is_durable_no_overwrite_and_nonzero_cli(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(workflow,"_run_block",side_effect=TimeoutError("bounded")):
            result=workflow.run_study(config(),temp)
            self.assertEqual(result["status"],"failed")
            self.assertFalse(result["performance_evidence"])
            with self.assertRaises(FileExistsError):workflow.run_study(config(),temp)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"config.json";path.write_text(json.dumps(config()))
            with patch.object(workflow,"run_study",return_value=dict(status="qualification_failed")):
                self.assertEqual(workflow.main(["--config",str(path),"--out",str(Path(temp)/"out")]),1)

    def test_full_matrix_is_validated_before_performance_claim(self):
        pristine=fixture()
        with tempfile.TemporaryDirectory() as temp,patch.object(workflow,"_run_block",side_effect=pristine["blocks"]):
            result=workflow.run_study(config(),temp)
            self.assertEqual(result["status"],"completed")
            self.assertTrue(result["performance_evidence"])
        pristine["blocks"][0]["ranks"][0]["trials"][0]["passed"]=False
        with tempfile.TemporaryDirectory() as temp,patch.object(workflow,"_run_block",side_effect=pristine["blocks"]):
            result=workflow.run_study(config(),temp)
            self.assertEqual(result["status"],"qualification_failed")
            self.assertFalse(result["performance_evidence"])


if __name__ == "__main__":
    unittest.main()
