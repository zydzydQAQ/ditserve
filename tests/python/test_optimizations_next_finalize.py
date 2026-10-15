"""CPU fakes and temporary metadata only; never inspect real study results."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.optimizations import next_finalize as target


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.output,self.ledger=self.root/"study/run01",self.root/"study/gpu-budget.json"
        self.config_path,self.auth_path=self.root/"config.json",self.root/"authorization.json"
        self.config=dict(frozen="temporary CPU fixture")
        atomic_json(self.config_path,self.config)
        atomic_json(self.auth_path,dict(scope="serving_next_validation",approved_by_user=True,
            budget_unlimited=True,authorized_gpu_seconds=None))
        self.auth=dict(path=str(self.auth_path),sha256=file_digest(self.auth_path))
        self.pid,self.started=912345,1000.
        self.state=dict(pid=self.pid,start_ticks=100,boot_id="temporary-boot",started_unix=999.,alive=False)
        self.chain=dict(pid=self.pid,started_unix=self.started,output=str(self.output),ledger=str(self.ledger),
            config=dict(path=str(self.config_path),sha256=file_digest(self.config_path)),config_sha256=digest(self.config),
            authorization=self.auth,status="completed",phases=[])
        self.budget=dict(active=None,sessions=[],used_gpu_seconds=80.)
        self.calls,self.sleeps,self.now=[],[],0.
        for phase in target.PHASES:
            attempt=self.output/phase/"attempt00"
            identity=dict(phase=phase,output=str(attempt),run_root=str(self.output),config_sha256=digest(self.config),
                authorization_source=self.auth,source_sha256="a"*64)
            charge=dict(session_id=phase,stage=phase,identity=identity,status="completed",gpu_count=1,gpu_seconds=10.)
            self.budget["sessions"].append(charge)
            evidence=attempt/target.NATIVE.get(phase,phase+".json")
            atomic_json(evidence,dict(status="completed",rows=[]))
            atomic_json(attempt/"config.json",self.config)
            atomic_json(attempt/"study.json",dict(temporary=True))
            terminal=dict(status="completed",returncode=0,charge=charge,output=str(attempt),evidence_path=str(evidence),
                config_sha256=digest(self.config),authorization_source=self.auth,
                final_occupancy=dict(status="exclusive_observed",owned=[],foreign=[]))
            atomic_json(attempt/"supervisor.json",terminal)
            path=self.output/phase/"terminal.json"
            atomic_json(path,terminal)
            self.chain["phases"].append(dict(phase=phase,status="completed",session_id=phase,
                terminal=dict(path=str(path),sha256=file_digest(path))))
        atomic_json(self.output/"chain.json",self.chain)
        atomic_json(self.ledger,self.budget)
        for name in ("study.json","protocol.json","workloads.json"):
            atomic_json(self.output/name,dict(temporary=True))
        atomic_json(self.output/"audits/final-integrity.json",dict(status="failed",error="stale audit fixture"))

    def report(self,output,ledger):
        self.calls.append("report")
        self.assertIsNone(read_json(ledger)["active"])
        atomic_json(output/"report.json",dict(temporary=True))
        atomic_json(output/"excel-verification.json",dict(status="verified"))
        (output/"results.xlsx").write_bytes(b"CPU fake workbook")
        (output/"REPORT.md").write_text("CPU fake report\n")
        return dict(status="recorded",report=str(output/"report.json"))

    def audit(self,output,ledger):
        self.calls.append("audit")
        value=dict(status="passed",all_phases_terminal=True)
        atomic_json(output/"audits/final-integrity.json",value)
        return value

    def sleep(self,seconds):
        self.sleeps.append(seconds)
        self.now+=seconds

    def run_finalizer(self,**overrides):
        options=dict(chain_pid=self.pid,chain_started_unix=self.started,config_path=self.config_path,
            output=self.output,ledger=self.ledger,authorization_path=self.auth_path,
            process_fn=lambda pid:copy.deepcopy(self.state),sleep=self.sleep,clock=lambda:self.now,
            report_fn=self.report,audit_fn=self.audit)
        options.update(overrides)
        return target.finalize_after_chain(**options)

    def change(self,path,callback):
        value=read_json(path)
        callback(value)
        atomic_json(path,value)

    def test_fresh_report_audit_preserves_chain_ledger_native_and_previous_audit(self):
        protected={p:p.read_bytes() for p in self.output.rglob("*.json") if p.name!="final-integrity.json"}
        protected[self.ledger]=self.ledger.read_bytes()
        previous=(self.output/"audits/final-integrity.json").read_bytes()
        value=self.run_finalizer()
        self.assertEqual(value["status"],"completed",value.get("reason"))
        self.assertEqual(self.calls,["report","audit"])
        self.assertEqual(Path(value["previous_audit"]["preserved_copy"]["path"]).read_bytes(),previous)
        self.assertEqual(value["previous_audit"]["source"]["sha256"],value["previous_audit"]["preserved_copy"]["sha256"])
        for path,raw in protected.items():
            self.assertEqual(path.read_bytes(),raw)
        self.assertFalse(value["gpu_used"])
        self.assertEqual(value["phases_executed"],0)
        self.assertFalse(value["raw_exported"])
        self.assertEqual(read_json(value["receipt"]),value)

    def test_waits_for_exact_chain_exit_then_accepts_only_stale_final_audit_failure(self):
        self.state["alive"]=True
        self.change(self.output/"chain.json",lambda d:d.update(status="running"))
        def sleep(seconds):
            self.sleep(seconds)
            self.assertFalse(self.calls)
            if len(self.sleeps)==2:
                self.state["alive"]=False
                self.change(self.output/"chain.json",lambda d:d.update(status="needs_attention",reason=target.AUDIT_FAILURE))
        value=self.run_finalizer(sleep=sleep)
        self.assertEqual(value["status"],"completed",value.get("reason"))
        self.assertEqual(self.sleeps,[5.,5.])
        self.assertEqual(read_json(self.output/"chain.json")["status"],"needs_attention")

    def test_missing_failed_or_unclean_phase_is_not_retried(self):
        path=self.output/"sana_graph/terminal.json"
        original=path.read_bytes()
        path.unlink()
        value=self.run_finalizer()
        self.assertEqual(value["status"],"failed")
        self.assertFalse(self.calls)
        path.write_bytes(original)
        for change in (lambda d:d.update(status="failed",returncode=1),
                       lambda d:d["final_occupancy"].update(owned=[dict(pid=1)])):
            self.change(path,change)
            value=self.run_finalizer()
            self.assertEqual(value["status"],"failed")
            self.assertFalse(self.calls)
            path.write_bytes(original)

    def test_occupied_chain_lock_does_not_write_report_or_replace_audit(self):
        previous=(self.output/"audits/final-integrity.json").read_bytes()
        with (self.output/"chain.lock").open("a+") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            value=self.run_finalizer()
        self.assertEqual(value["status"],"failed")
        self.assertIn("lock is occupied",value["reason"])
        self.assertFalse(self.calls)
        self.assertEqual((self.output/"audits/final-integrity.json").read_bytes(),previous)

    def test_wait_rejects_changed_chain_config_or_authorization_identity(self):
        self.state["alive"]=True
        for path,change in ((self.output/"chain.json",lambda d:d.update(pid=self.pid+1)),
                            (self.config_path,lambda d:d.update(frozen="changed")),
                            (self.auth_path,lambda d:d.update(approved_by_user=False))):
            original=path.read_bytes()
            def sleep(seconds):
                self.sleep(seconds)
                self.change(path,change)
            value=self.run_finalizer(sleep=sleep)
            self.assertEqual(value["status"],"failed")
            self.assertFalse(self.calls)
            path.write_bytes(original)

    def test_pid_reuse_before_or_during_wait_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"start identity"):
            self.run_finalizer(process_start_ticks=101,boot_id="temporary-boot")
        self.state["started_unix"]=self.started+10.
        with self.assertRaisesRegex(ValueError,"reused"):
            self.run_finalizer()
        self.state.update(started_unix=999.,alive=True)
        def sleep(seconds):
            self.sleep(seconds)
            self.state["start_ticks"]+=1
        value=self.run_finalizer(sleep=sleep)
        self.assertEqual(value["status"],"failed")
        self.assertIn("PID was reused",value["reason"])
        self.assertFalse(self.calls)

    def test_new_active_lease_prevents_refresh_without_ledger_repair(self):
        self.change(self.ledger,lambda d:d.update(active=dict(session_id="new-owner")))
        before=self.ledger.read_bytes()
        value=self.run_finalizer()
        self.assertEqual(value["status"],"failed")
        self.assertIn("still active",value["reason"])
        self.assertEqual(self.ledger.read_bytes(),before)
        self.assertFalse(self.calls)

    def test_metadata_change_during_report_stops_before_audit(self):
        def report(output,ledger):
            result=self.report(output,ledger)
            self.change(ledger,lambda d:d.update(active=dict(session_id="new-owner")))
            return result
        value=self.run_finalizer(report_fn=report)
        self.assertEqual(value["status"],"failed")
        self.assertIn("protected metadata changed",value["reason"])
        self.assertEqual(self.calls,["report"])
        self.assertEqual(read_json(self.ledger)["active"]["session_id"],"new-owner")

    def test_generic_chain_error_and_dead_running_chain_are_not_accepted(self):
        for status,reason in (("needs_attention","RuntimeError: phase failed or incomplete; no automatic retry"),
                              ("running",None)):
            self.change(self.output/"chain.json",lambda d:d.update(status=status,reason=reason))
            with self.assertRaisesRegex(ValueError,"allowed terminal"):
                self.run_finalizer()
        self.assertFalse(self.calls)

    def test_failed_fresh_audit_remains_failed_and_preserves_prior_copy(self):
        def fail(output,ledger):
            self.calls.append("audit")
            value=dict(status="failed",all_phases_terminal=False,error="independent mismatch")
            atomic_json(output/"audits/final-integrity.json",value)
            return value
        before=(self.output/"chain.json").read_bytes()
        value=self.run_finalizer(audit_fn=fail)
        self.assertEqual(value["status"],"failed")
        self.assertEqual(value["audit_result"]["reason"],"independent mismatch")
        self.assertTrue(Path(value["previous_audit"]["preserved_copy"]["path"]).is_file())
        self.assertEqual((self.output/"chain.json").read_bytes(),before)

    def test_timeout_is_cpu_only_and_poll_is_bounded(self):
        self.state["alive"]=True
        value=self.run_finalizer(wait_seconds=12.,poll_seconds=10.)
        self.assertEqual(value["status"],"failed")
        self.assertIn("timed out",value["reason"])
        self.assertEqual(self.sleeps,[10.,2.])
        self.assertFalse(self.calls)
        with self.assertRaisesRegex(ValueError,"at most 10"):
            self.run_finalizer(poll_seconds=11.)

    def test_symlinked_prior_audit_is_never_copied_or_overwritten(self):
        path=self.output/"audits/final-integrity.json"
        path.unlink()
        target_path=self.root/"external.json"
        target_path.write_text('{"keep":true}\n')
        path.symlink_to(target_path)
        before=target_path.read_bytes()
        value=self.run_finalizer()
        self.assertEqual(value["status"],"failed")
        self.assertEqual(target_path.read_bytes(),before)
        self.assertFalse(self.calls)

    def test_fresh_import_does_not_load_frameworks_or_execution_or_reporting(self):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(Path(target.__file__).resolve().parents[2]))
        script="import sys; import ditserve.optimizations.next_finalize; assert not any(k in sys.modules for k in ('torch','numpy','ditserve.optimizations.next_study','ditserve.optimizations.next_audit','ditserve.optimizations.next_report','ditserve.optimizations.next_retention'))"
        result=subprocess.run([sys.executable,"-c",script],env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_process_start_ticks_and_zombie_exit_are_bound(self):
        fields=["Z"]+["0"]*18+["12345"]
        def read(path):
            if str(path).endswith("/912345/stat"):
                return "912345 (name (nested)) "+" ".join(fields)
            if str(path).endswith("boot_id"):
                return "fixed-boot\n"
            return "cpu 0\nbtime 100\n"
        with patch.object(target.Path,"read_text",read),patch.object(target.os,"sysconf",return_value=100):
            value=target._process(self.pid)
        self.assertEqual(value,dict(pid=self.pid,start_ticks=12345,boot_id="fixed-boot",alive=False,started_unix=223.45))


if __name__=="__main__":
    unittest.main()
