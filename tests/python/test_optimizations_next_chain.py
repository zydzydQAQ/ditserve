"""CPU-only temporary-ledger tests; every phase executor is a fake."""

import copy
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.optimizations import next_chain as target


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output, self.ledger = self.root / "study/run01", self.root / "study/gpu-budget.json"
        self.config_path, self.auth_path = self.root / "config.json", self.root / "authorization.json"
        self.config = dict(frozen="CPU fixture")
        atomic_json(self.config_path, self.config)
        atomic_json(self.auth_path, dict(approved_by_user=True, budget_unlimited=True))
        self.auth = dict(path=str(self.auth_path), sha256=file_digest(self.auth_path))
        self.now, self.executions, self.reports, self.audits, self.sleeps = 100., [], [], [], []
        self.alive, self.audit_status = {}, "passed"
        self.budget = dict(active=None, sessions=[], used_gpu_seconds=0., stage_used_gpu_seconds={})
        atomic_json(self.ledger, self.budget)
        self.phases = ("diagnosis", "sana_vae")
        patcher = patch.object(target, "PHASES", self.phases)
        patcher.start()
        self.addCleanup(patcher.stop)

    def session(self, phase, session_id=None):
        return dict(session_id=session_id or phase, stage=phase, supervisor_pid=32101, gpu_count=1,
            started_monotonic=50., identity=dict(phase=phase, run_root=str(self.output),
                output=str(self.output / phase / "attempt00"), config_sha256=digest(self.config),
                authorization_source=self.auth, source_sha256="0"*64))

    def finish(self, phase, status="completed", session=None, publish=True):
        charge = dict(session or self.session(phase), status="failed" if status in ("failed", "incomplete") else "completed", gpu_seconds=10.)
        value = dict(status=status, decision="fixture", reason="scientific gate rejected" if status == "stopped" else "fixture",
            returncode=0 if charge["status"] == "completed" else 1, charge=charge,
            output=charge["identity"]["output"], config_sha256=digest(self.config), authorization_source=self.auth,
            final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[]))
        budget = read_json(self.ledger)
        budget.update(active=None)
        budget["sessions"].append(charge)
        budget["used_gpu_seconds"] += 10.
        budget["stage_used_gpu_seconds"][phase] = budget["stage_used_gpu_seconds"].get(phase, 0.)+10.
        atomic_json(self.ledger, budget)
        if publish:
            self.publish(phase, value)
        return value

    def publish(self, phase, value):
        atomic_json(Path(value["output"]) / "supervisor.json", value)
        atomic_json(self.output / phase / "terminal.json", value)

    def execute(self, config, output, ledger, phase, authorization, **options):
        self.assertIsNone(read_json(ledger)["active"])
        self.assertEqual(options, dict(wait_idle=3600.))
        self.assertEqual((config, output, ledger, authorization), (self.config, self.output, self.ledger, self.auth_path))
        self.executions.append(phase)
        return self.finish(phase)

    def report(self, output, ledger):
        self.assertIsNone(read_json(ledger)["active"])
        self.reports.append(tuple(self.executions))
        return dict(status="recorded", partial=True, report=str(output / "report.json"), requests=0)

    def audit(self, output, ledger):
        self.assertIsNone(read_json(ledger)["active"])
        self.audits.append(True)
        value = dict(status=self.audit_status)
        atomic_json(output / "audits/final-integrity.json", value)
        return value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def run_chain(self, **overrides):
        options = dict(execute_fn=self.execute, report_fn=self.report, audit_fn=self.audit,
            config_loader=lambda path:self.config,
            authorization_loader=lambda path:({}, self.auth),
            clock=lambda:self.now, sleep=self.sleep, pid_alive=lambda pid:self.alive.get(pid, False))
        options.update(overrides)
        return target.run_chain(self.config_path, self.output, self.ledger, self.auth_path, **options)

    def test_serial_execution_reports_each_settlement_audits_and_never_prunes(self):
        raw = self.output / "retained.raw"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"root analysis dependency")
        with patch("ditserve.compact_results.prune", side_effect=AssertionError("no pruning")):
            result = self.run_chain()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.executions, list(self.phases))
        self.assertEqual(len(self.reports), 3)
        self.assertEqual(len(self.audits), 1)
        self.assertEqual(result["costs"]["settled_gpu_seconds"], 20.)
        self.assertFalse(result["costs"]["chain_owns_extra_gpu_lease"])
        self.assertEqual(raw.read_bytes(), b"root analysis dependency")
        self.assertFalse(result["cleanup_performed"])

    def test_scientific_stopped_phase_is_skipped_without_retry(self):
        self.finish("diagnosis", "stopped")
        result = self.run_chain()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.executions, ["sana_vae"])
        self.assertEqual(result["phases"][0]["action"], "skipped_existing")

    def test_failed_existing_phase_stops_without_gpu_retry(self):
        self.finish("diagnosis", "failed")
        before = self.ledger.read_bytes()
        result = self.run_chain()
        self.assertEqual(result["status"], "needs_attention")
        self.assertIn("no automatic retry", result["reason"])
        self.assertEqual(self.executions, [])
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse(self.audits)

    def test_new_phase_failure_or_resource_block_stops_chain(self):
        def failed(config, output, ledger, phase, auth, **kwargs):
            self.executions.append(phase)
            return self.finish(phase, "incomplete")
        result = self.run_chain(execute_fn=failed)
        self.assertEqual(result["status"], "needs_attention")
        self.assertEqual(self.executions, ["diagnosis"])
        self.assertFalse(self.audits)

    def test_preflight_block_preserves_ledger_without_creating_terminal(self):
        before = self.ledger.read_bytes()
        result = self.run_chain(execute_fn=lambda *a, **kw:dict(status="blocked", reason="foreign GPU owner"))
        self.assertEqual(result["status"], "needs_attention")
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse((self.output / "diagnosis/terminal.json").exists())
        self.assertFalse(self.audits)

    def test_explicit_session_waits_for_settlement_publication_and_supervisor_exit(self):
        session = self.session("diagnosis", "owned-session")
        self.budget["active"] = session
        atomic_json(self.ledger, self.budget)
        self.alive[32101] = True
        pending = []
        def sleep(seconds):
            self.sleep(seconds)
            if len(self.sleeps) == 1:
                pending.append(self.finish("diagnosis", session=session, publish=False))
            elif len(self.sleeps) == 2:
                self.publish("diagnosis", pending[0])
            elif len(self.sleeps) == 3:
                self.alive[32101] = False
        result = self.run_chain(wait_session_id="owned-session", sleep=sleep)
        self.assertEqual(result["status"], "completed", result.get("reason"))
        self.assertEqual(self.sleeps, [5., 1., 1.])
        self.assertEqual(self.executions, ["sana_vae"])
        self.assertEqual(result["waited_terminal"]["path"], str(self.output / "diagnosis/terminal.json"))

    def test_wait_rejects_unknown_or_dead_active_supervisor_without_ledger_repair(self):
        session = self.session("diagnosis", "owned-session")
        self.budget["active"] = session
        atomic_json(self.ledger, self.budget)
        before = self.ledger.read_bytes()
        for session_id in (None, "wrong-session", "owned-session"):
            with self.subTest(session_id=session_id):
                result = self.run_chain(wait_session_id=session_id)
                self.assertEqual(result["status"], "needs_attention")
                self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse(self.executions)
        self.assertFalse(self.sleeps)

    def test_changed_wait_identity_and_terminal_gap_fail_closed(self):
        session = self.session("diagnosis", "owned-session")
        session["identity"]["config_sha256"] = "1"*64
        self.budget["active"] = session
        atomic_json(self.ledger, self.budget)
        self.alive[32101] = True
        result = self.run_chain(wait_session_id="owned-session")
        self.assertIn("configuration", result["reason"])
        self.assertFalse(self.sleeps)
        session["identity"]["config_sha256"] = digest(self.config)
        atomic_json(self.ledger, dict(self.budget, active=None))
        self.finish("diagnosis", session=session, publish=False)
        self.alive[32101] = False
        result = self.run_chain(wait_session_id="owned-session")
        self.assertIn("within 10 seconds", result["reason"])
        self.assertEqual(sum(self.sleeps), 10.)
        self.assertFalse(self.executions)

    def test_active_allocation_appearing_between_stages_prevents_overlap(self):
        def report(output, ledger):
            value = self.report(output, ledger)
            budget = read_json(ledger)
            budget["active"] = self.session("sana_vae", "foreign-session")
            atomic_json(ledger, budget)
            return value
        result = self.run_chain(report_fn=report)
        self.assertEqual(result["status"], "needs_attention")
        self.assertEqual(self.executions, ["diagnosis"])
        self.assertEqual(read_json(self.ledger)["active"]["session_id"], "foreign-session")

    def test_chain_lock_and_final_audit_failure_do_not_start_another_driver(self):
        self.output.mkdir(parents=True)
        with (self.output / "chain.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "another chain"):
                self.run_chain()
        self.assertFalse(self.executions)
        self.audit_status = "failed"
        result = self.run_chain()
        self.assertEqual(result["status"], "needs_attention")
        self.assertIn("integrity audit", result["reason"])

    def test_zombie_supervisor_is_exited_even_before_launcher_reaps_it(self):
        with patch.object(target.os, "kill", return_value=None) as probe, \
             patch.object(target.Path, "read_text", return_value="32101 (name (nested)) Z 1 2 3"):
            self.assertFalse(target._pid_alive(32101))
            probe.assert_called_once_with(32101, 0)
        with patch.object(target.os, "kill", return_value=None), \
             patch.object(target.Path, "read_text", return_value="32101 (normal) S 1 2 3"):
            self.assertTrue(target._pid_alive(32101))


if __name__ == "__main__":
    unittest.main()
