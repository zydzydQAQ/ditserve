"""Temporary metadata and a fake executor; never loads models or touches GPUs."""
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.optimizations import next_numerics_retry as target
from ditserve.optimizations.next_protocol import PHASES


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.parent,self.parent_ledger=self.root/"main/run01",self.root/"main/gpu-budget.json"
        self.output,self.ledger=self.root/"retry/run01",self.root/"retry/gpu-budget.json"
        self.config_path,self.auth_path=self.root/"config.json",self.root/"authorization.json"
        self.config=dict(frozen="CPU fixture")
        atomic_json(self.config_path,self.config)
        atomic_json(self.auth_path,dict(scope="serving_next_validation",approved_by_user=True,budget_unlimited=True,
            authorized_gpu_seconds=None,user_instruction="就按照这个继续；取消8小时GPU小时硬上限，GPU无上限，开始在线对照；继续运行"))
        self.auth=target._source(self.auth_path)
        self.chain_pid,self.finalizer_pid,self.started=912340,912341,1000.
        self.identities=dict(chain=dict(pid=self.chain_pid,start_ticks=100,boot_id="temporary-boot"),
            finalizer=dict(pid=self.finalizer_pid,start_ticks=200,boot_id="temporary-boot"))
        self.states={row["pid"]:dict(row,alive=False,started_unix=999.) for row in self.identities.values()}
        self.chain=dict(pid=self.chain_pid,started_unix=self.started,output=str(self.parent),ledger=str(self.parent_ledger),
            config=target._source(self.config_path),config_sha256=digest(self.config),authorization=self.auth,status="completed",phases=[])
        budget=dict(active=None,sessions=[],used_gpu_seconds=80.,limit_gpu_seconds=None,budget_unlimited=True)
        for phase in PHASES:
            attempt=self.parent/phase/"attempt00"
            identity=dict(phase=phase,output=str(attempt),run_root=str(self.parent),config_sha256=digest(self.config),
                authorization_source=self.auth,source_sha256="a"*64)
            charge=dict(session_id=phase,stage=phase,identity=identity,status="completed",gpu_count=1,gpu_seconds=10.)
            budget["sessions"].append(charge)
            names=dict(diagnosis="diagnosis.json",sana_vae="sana-candidate.json",sana_graph="sana-candidate.json",
                sana_numerics="sana-numerics.json",conditional="conditional.json")
            path=attempt/names.get(phase,phase+".json")
            native=dict(status="completed",rows=[])
            if phase=="sana_numerics":
                native=dict(status="passed",failed_units=8,compared_images=0,rows=[dict(unit_index=i,
                    status="failed" if i<4 or i>=8 else "completed",comparisons=[],
                    error="RuntimeError: self.stride(-1) must be 1 to view Float as Byte (different element sizes), but got 0")
                    for i in range(12)])
            atomic_json(path,native)
            atomic_json(attempt/"config.json",self.config)
            atomic_json(attempt/"study.json",dict(fixture=True))
            terminal=dict(status="passed" if phase=="sana_numerics" else "completed",returncode=0,charge=charge,
                output=str(attempt),evidence_path=str(path),config_sha256=digest(self.config),authorization_source=self.auth,
                final_occupancy=dict(status="exclusive_observed",owned=[],foreign=[]))
            atomic_json(attempt/"supervisor.json",terminal)
            terminal_path=self.parent/phase/"terminal.json"
            atomic_json(terminal_path,terminal)
            self.chain["phases"].append(dict(phase=phase,status=terminal["status"],session_id=phase,
                terminal=target._source(terminal_path)))
        atomic_json(self.parent_ledger,budget)
        atomic_json(self.parent/"chain.json",self.chain)
        for name in ("study.json","protocol.json","workloads.json"):
            atomic_json(self.parent/name,dict(fixture=True))
        atomic_json(self.parent/"report.json",dict(fixture=True))
        audit=dict(status="passed",all_phases_terminal=True,output=str(self.parent),ledger=str(self.parent_ledger),
            ledger_source=target._read(self.parent_ledger)[1],report_sha256=file_digest(self.parent/"report.json"))
        atomic_json(self.parent/"audits/final-integrity.json",audit)
        self.receipt=self.parent/"audits/finalize-attempt000/receipt.json"
        atomic_json(self.receipt,dict(status="completed",pid=self.finalizer_pid,process_identity=self.identities["chain"],
            chain_identity=target._identity(self.chain),chain_final=target._read(self.parent/"chain.json")[1],
            audit=target._read(self.parent/"audits/final-integrity.json")[1]))
        self.parents=[]
        for index in range(3):
            path=self.root/("old%d.json"%index)
            atomic_json(path,dict(active=None,limit_gpu_seconds=None,budget_unlimited=True,used_gpu_seconds=10.))
            self.parents.append(path)
        self.parent_patch=patch.object(target,"PARENTS",tuple(self.parents))
        self.parent_patch.start()
        self.addCleanup(self.parent_patch.stop)
        self.prepared=dict(config_sha256=digest(self.config),historical_ledgers=[target._source(p) for p in self.parents],
            cache_local_binding=dict(mathematical_source_sha256="b"*64,mathematical_source_files={"original.py":"c"*64},
                bindings=dict(original=dict(path="/unchanged/cache-quality.json",sha256="d"*64))))
        self.calls,self.sleeps,self.now=[],[],0.

    def factory(self,prepared,parents,implementation):
        self.assertEqual(tuple(parents),tuple(self.parents)+(self.parent_ledger,))
        self.assertEqual(prepared["historical_ledgers"],self.prepared["historical_ledgers"]+[target._source(self.parent_ledger)])
        original=self.prepared["cache_local_binding"]
        for name in ("mathematical_source_files","mathematical_source_sha256"):
            self.assertEqual(prepared["cache_local_binding"][name],original[name])
        self.assertEqual(prepared["cache_local_binding"]["bindings"]["original"],original["bindings"]["original"])
        self.assertEqual(prepared["cache_local_binding"]["bindings"]["numerics_retry_admission"],target._source(self.output/"retry-binding.json"))
        def execute(config,output,ledger,phase,authorization,*,wait_idle):
            self.calls.append("execute")
            self.assertEqual(phase,"sana_numerics")
            self.assertTrue(all(not state["alive"] for state in self.states.values()))
            self.assertEqual(read_json(output/"retry-binding.json")["required_units"],12)
            value=dict(status="passed",returncode=0,output=str(output/phase/"attempt00"),charge=dict(status="completed",gpu_seconds=7.))
            atomic_json(ledger,dict(active=None,sessions=[value["charge"]],parents={str(p):target._source(p) for p in parents}))
            atomic_json(output/phase/"terminal.json",value)
            return value
        return execute

    def report(self,output,ledger):
        self.calls.append("report")
        return dict(status="completed",integrity_valid=True,diagnostic_complete=True,verified_comparisons=12,online_qualified=False)

    def sleep(self,seconds):
        self.sleeps.append(seconds)
        self.now+=seconds

    def run_retry(self,**overrides):
        options=dict(chain_pid=self.chain_pid,chain_started_unix=self.started,chain_start_ticks=100,
            finalizer_pid=self.finalizer_pid,finalizer_start_ticks=200,boot_id="temporary-boot",
            parent_output=self.parent,parent_ledger=self.parent_ledger,finalizer_receipt=self.receipt,
            output=self.output,ledger=self.ledger,config_path=self.config_path,authorization_path=self.auth_path,
            config_loader=lambda path:self.config,prepare_fn=lambda *a,**kw:copy.deepcopy(self.prepared),
            executor_factory=self.factory,report_fn=self.report,process_fn=lambda pid:copy.deepcopy(self.states.get(pid)),
            sleep=self.sleep,clock=lambda:self.now)
        options.update(overrides)
        return target.run_retry(**options)

    def test_single_replay_binds_four_parents_and_preserves_every_old_file(self):
        before={p:p.read_bytes() for p in self.parent.rglob("*.json")}
        before[self.parent_ledger]=self.parent_ledger.read_bytes()
        value=self.run_retry()
        self.assertEqual(value["status"],"completed",value.get("reason"))
        self.assertEqual(self.calls,["execute","report"])
        admission=read_json(self.output/"retry-binding.json")
        self.assertEqual(len(admission["parents"]),4)
        self.assertEqual((admission["required_units"],admission["required_generated_images"],admission["required_comparisons"]),(12,16,12))
        self.assertEqual(admission["scope"],target.PROVENANCE_SCOPE)
        self.assertFalse(value["changes_serving_default"])
        for path,raw in before.items():
            self.assertEqual(path.read_bytes(),raw)
        with self.assertRaisesRegex(ValueError,"already used"):
            self.run_retry()

    def test_waits_for_both_processes_without_gpu_or_ledger(self):
        for state in self.states.values():
            state["alive"]=True
        def sleep(seconds):
            self.sleep(seconds)
            self.assertFalse(self.calls)
            self.assertFalse(self.ledger.exists())
            self.states[self.chain_pid if len(self.sleeps)==1 else self.finalizer_pid]["alive"]=False
        value=self.run_retry(sleep=sleep)
        self.assertEqual(value["status"],"completed",value.get("reason"))
        self.assertEqual(self.sleeps,[5.,5.])

    def test_private_function_copy_preserves_original_globals_and_keyword_defaults(self):
        from ditserve.optimizations import next_study
        original_phases,original_parents=next_study.PHASES,next_study.PARENTS
        def fake(config,*,observe=False):
            return PHASES,PARENTS,inspect_study(config,observe=observe)
        # These global names intentionally belong only to the copied namespace.
        with patch.object(next_study,"execute",fake):
            private=target._isolated_executor(self.prepared,self.parents,{})
            phases,parents,prepared=private(self.config)
        self.assertEqual(phases,("sana_numerics",))
        self.assertEqual(parents,tuple(self.parents))
        self.assertEqual(prepared,self.prepared)
        self.assertEqual(fake.__kwdefaults__,dict(observe=False))
        self.assertEqual((next_study.PHASES,next_study.PARENTS),(original_phases,original_parents))

    def test_pid_reuse_or_changed_chain_identity_prevents_admission(self):
        self.states[self.finalizer_pid]["start_ticks"]+=1
        value=self.run_retry()
        self.assertEqual(value["status"],"failed")
        self.assertIn("PID was reused",value["reason"])
        self.assertFalse(self.calls)

    def test_parent_lock_active_ledger_and_incomplete_finalizer_are_fail_closed(self):
        with (self.parent/"chain.lock").open("a+") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            value=self.run_retry()
        self.assertEqual(value["status"],"failed")
        self.assertIn("lock remains occupied",value["reason"])
        self.assertFalse(self.calls)

    def test_live_parent_ledger_is_never_frozen_or_repaired(self):
        budget=read_json(self.parent_ledger)
        budget["active"]=dict(session_id="still-owned")
        atomic_json(self.parent_ledger,budget)
        before=self.parent_ledger.read_bytes()
        value=self.run_retry()
        self.assertEqual(value["status"],"failed")
        self.assertFalse((self.output/"retry-binding.json").exists())
        self.assertEqual(self.parent_ledger.read_bytes(),before)
        self.assertFalse(self.calls)

    def test_unsuccessful_finalizer_blocks_even_with_terminal_chain(self):
        receipt=read_json(self.receipt)
        receipt["status"]="failed"
        atomic_json(self.receipt,receipt)
        value=self.run_retry()
        self.assertEqual(value["status"],"failed")
        self.assertIn("fresh successful",value["reason"])
        self.assertFalse(self.calls)

    def test_failure_after_execution_is_reported_once_without_gpu_retry(self):
        def factory(*args):
            execute=self.factory(*args)
            def failed(*values,**options):
                result=execute(*values,**options)
                result.update(status="incomplete",returncode=1)
                return result
            return failed
        value=self.run_retry(executor_factory=factory)
        self.assertEqual(value["status"],"unverified")
        self.assertEqual(self.calls,["execute","report"])

    def test_parent_mutation_after_child_is_detected_before_qualification(self):
        def factory(*args):
            execute=self.factory(*args)
            def changed(*values,**options):
                result=execute(*values,**options)
                budget=read_json(self.parent_ledger)
                budget["used_gpu_seconds"]+=1.
                atomic_json(self.parent_ledger,budget)
                return result
            return changed
        value=self.run_retry(executor_factory=factory)
        self.assertEqual(value["status"],"failed")
        self.assertIn("protected metadata changed",value["reason"])
        self.assertEqual(self.calls,["execute","report"])

    def test_incomplete_independent_report_never_becomes_completed_driver(self):
        def partial(output,ledger):
            self.calls.append("report")
            return dict(status="partial",integrity_valid=True,diagnostic_complete=False,verified_comparisons=8)
        value=self.run_retry(report_fn=partial)
        self.assertEqual(value["status"],"unverified")
        self.assertTrue(value["execution_finished"])
        self.assertFalse(value["diagnostic_complete"])
        self.assertEqual(self.calls,["execute","report"])

    def test_existing_allocation_accepts_four_immutable_parents_and_one_phase(self):
        from ditserve.optimizations.next_budget import NextAllocation
        ticks=iter((100.,113.))
        identity=dict(phase="sana_numerics",config_sha256=digest(self.config),source_sha256="b"*64,
            output=str(self.output/"sana_numerics/attempt00"),run_root=str(self.output),
            gpu_uuids=["GPU-fixture"],authorization_source=self.auth)
        with NextAllocation(self.ledger,parents=self.parents+[self.parent_ledger],authorization_binding=self.auth,
                gpu_uuid="GPU-fixture",stages=("sana_numerics",),monotonic=lambda:next(ticks),clock=lambda:1000.) as budget:
            budget.begin("sana_numerics",gpu_count=1,identity=identity)
            budget.child_started(os.getpid()+1000)
            row=budget.finish("completed")
        self.assertEqual(row["gpu_seconds"],13.)
        stored=read_json(self.ledger)
        self.assertEqual(stored["stages"],["sana_numerics"])
        self.assertEqual(len(stored["parents"]),4)
        self.assertEqual(stored["parents"][str(self.parent_ledger)]["sha256"],file_digest(self.parent_ledger))

    def test_wait_timeout_never_creates_budget_or_gpu_work(self):
        self.states[self.finalizer_pid]["alive"]=True
        value=self.run_retry(wait_seconds=6.)
        self.assertEqual(value["status"],"failed")
        self.assertEqual(self.sleeps,[5.,1.])
        self.assertFalse(self.calls)
        self.assertFalse(self.ledger.exists())

    def test_fresh_import_is_cpu_only_and_does_not_load_supervisor(self):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(Path(target.__file__).resolve().parents[2]))
        script="import sys; import ditserve.optimizations.next_numerics_retry; assert not any(k in sys.modules for k in ('torch','numpy','ditserve.optimizations.next_study','ditserve.optimizations.next_numerics'))"
        result=subprocess.run([sys.executable,"-c",script],env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=="__main__":
    unittest.main()
