"""CPU-only supervision checks with actual freezing/accounting and a fake child."""

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ditserve.directions import runner, protocol, budget as budget_module
from ditserve.directions.budget import Allocation, BudgetStop
from ditserve.directions.protocol import load_config
from ditserve.directions.worker import verify_admission
from ditserve.research.common import digest


class SupervisorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "run"
        self.ledger = self.root / "budget.json"
        self.parent = self.root / "historical.json"
        self.parent.write_text(json.dumps(dict(used_gpu_seconds=100., limit_gpu_seconds=28800.)))
        self.config = load_config()
        self.config["python"] = sys.executable
        self.elapsed = 1000.
        self.spawn = None
        self.tamper = False
        self.phase = "profile"
        # The parent is simulated just like the child. A container's test runner
        # may be PID 1, which the real orphan/admission guard correctly rejects.
        self.parent_pid = 424241

    def allocation(self, path, parent):
        return Allocation(path, parent, clock=lambda: 5000. + self.elapsed,
                          monotonic=lambda: self.elapsed)

    def popen(self, command, **kwargs):
        self.spawn = (command, kwargs)
        active = json.loads(self.ledger.read_text())["active"]
        self.assertIsNotNone(active)
        self.assertEqual(active["supervisor_pid"], self.parent_pid)
        self.assertEqual(command[command.index("--session-id") + 1], active["session_id"])
        self.assertTrue(kwargs["start_new_session"])
        self.elapsed += 3.  # Model/environment setup is charged.
        suite = self
        class Child:
            pid = 424242
            def wait(self, timeout):
                args = SimpleNamespace(ledger=suite.ledger, session_id=active["session_id"],
                    output=suite.output / "direction1", direction=1, phase=suite.phase)
                verify_admission(args, suite.config, getpid=lambda: self.pid,
                                 getpgrp=lambda: self.pid, getppid=lambda: suite.parent_pid,
                                 environ=kwargs["env"])
                (args.output / ("worker-%s.json" % suite.phase)).write_text(json.dumps(dict(
                    status="passed", eligible=True, cache_quality_qualified=True,
                    cache_performance_qualified=True)))
                suite.elapsed += 4.  # Compute and result persistence are charged.
                if suite.tamper:
                    target = Path(kwargs["env"]["PYTHONPATH"]) / "ditserve/directions/worker.py"
                    target.write_text(target.read_text() + "\n# unexpected mutation\n")
                return 0
        return Child()

    def cleanup(self, child):
        self.assertEqual(json.loads(self.ledger.read_text())["active"]["child_pgid"], child.pid)
        self.elapsed += 2.  # Cleanup remains inside the charge.
        return True

    def execute(self, *, final_status="exclusive_observed", cleanup_ok=True, reserve_for=None):
        def clean(child):
            self.cleanup(child)
            return cleanup_ok
        ready = dict(status="ready", models={name: dict(available=True) for name in self.config["models"]})
        observations = [dict(status="exclusive_observed", owned=[], foreign=[]),
                        dict(status=final_status, owned=[], foreign=[dict(pid=999)]
                             if final_status == "contended" else [])]
        with patch.object(runner, "PARENT_LEDGER", self.parent), \
             patch.object(budget_module, "os", SimpleNamespace(getpid=lambda: self.parent_pid)), \
             patch.object(runner, "Allocation", self.allocation), \
             patch.object(runner, "inspect", return_value=ready), \
             patch.object(runner.subprocess, "Popen", side_effect=self.popen), \
             patch.object(runner, "stop_owned_group", side_effect=clean), \
             patch.object(runner, "gpu_occupancy", side_effect=observations):
            return runner.execute(self.config, self.output, self.ledger, 1, self.phase,
                                  reserve_for=reserve_for)

    def test_normal_phase_runs_frozen_source_and_charges_through_cleanup(self):
        # Historical outputs are optional deployment assets. Supply the evidence
        # fixture explicitly instead of depending on the old machine's results.
        source = self.root / "source"
        for name in ("python", "configs", "tests/python"):
            shutil.copytree(runner.ROOT / name, source / name)
        historical = source / "results/research-pilot/run05/report.json"
        historical.parent.mkdir(parents=True)
        historical.write_text('{"status":"fixture"}\n')
        with patch.object(protocol, "ROOT", source):
            report = self.execute()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["eligible"])
        self.assertEqual(report["charge"]["gpu_seconds"], 9.)
        ledger = json.loads(self.ledger.read_text())
        self.assertIsNone(ledger["active"])
        self.assertEqual(ledger["stage_used_gpu_seconds"]["direction1"], 9.)
        _, spawn = self.spawn
        frozen = self.output / "sources/direction1-profile"
        self.assertEqual(spawn["env"]["PYTHONPATH"], str(frozen / "python"))
        self.assertEqual(spawn["env"]["CUDA_VISIBLE_DEVICES"], self.config["gpu_uuid"])
        self.assertTrue((frozen / "configs/research_cfg.json").is_file())
        self.assertTrue((frozen / "results/research-pilot/run05/report.json").is_file())
        # A real CPU Python import confirms __file__-derived ROOT follows the
        # snapshot and that D2's frozen catalog/implementation inputs resolve.
        probe = subprocess.run([sys.executable, "-c",
            "import json; from ditserve.directions import direction2 as d; "
            "from ditserve.directions import protocol as p; "
            "print(json.dumps({'root':str(d.ROOT),'protocol_root':str(p.ROOT),"
            "'dev':len(d._inputs('dev')),'identity':d._identity("
            "json.load(open(p.ROOT/'configs/research_directions.json')),"
            "d._settings(json.load(open(p.ROOT/'configs/research_directions.json'))))}))"],
            cwd=runner.ROOT, env=dict(spawn["env"], PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, check=True)
        imported = json.loads(probe.stdout)
        self.assertEqual(imported["root"], str(frozen))
        self.assertEqual(imported["protocol_root"], str(frozen))
        self.assertEqual(imported["dev"], 8)

    def test_late_foreign_job_invalidates_phase_and_blocks_next_phase(self):
        report = self.execute(final_status="contended")
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["eligible"])
        self.assertEqual(report["supervisor_failure"]["error_type"], "GPUIsolationError")
        self.assertEqual(report["charge"]["gpu_seconds"], 9.)
        with patch.object(runner, "inspect") as inspect:
            next_report = runner.execute(self.config, self.output, self.ledger, 1, "correctness")
        self.assertEqual(next_report["status"], "blocked")
        inspect.assert_not_called()

    def test_unconfirmed_group_cleanup_keeps_accounting_unsettled(self):
        with self.assertRaisesRegex(RuntimeError, "cannot confirm GPU cleanup"):
            self.execute(cleanup_ok=False)
        ledger = json.loads(self.ledger.read_text())
        self.assertIsNotNone(ledger["active"])
        self.assertEqual(ledger["sessions"], [])
        self.assertFalse((self.output / "direction1/supervisor-profile.json").exists())
        with self.assertRaises(BudgetStop):
            with self.allocation(self.ledger, self.parent):
                pass

    def test_frozen_source_mutation_invalidates_measurement(self):
        self.tamper = True
        report = self.execute()
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["eligible"])
        self.assertEqual(report["supervisor_failure"]["error_type"], "SourceIdentityError")

    def test_busy_admission_keeps_existing_ledger_unchanged(self):
        with self.allocation(self.ledger, self.parent):
            pass
        before = self.ledger.read_bytes()
        with patch.object(runner, "inspect", return_value=dict(status="blocked")), \
             patch.object(runner.subprocess, "Popen") as popen:
            report = runner.execute(self.config, self.output, self.ledger, 1, "profile")
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["gpu_work_started"])
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse((self.output / "sources").exists())
        popen.assert_not_called()

    def test_valid_repair_retry_charges_only_reserve(self):
        failed = self.root / "old-supervisor-profile.json"
        failed.write_text(json.dumps(dict(status="failed", direction=1, phase="profile",
            config_sha256=digest(self.config), charge=dict(gpu_seconds=1.))))
        report = self.execute(reserve_for=failed)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["charge"]["stage"], "reserve")
        ledger = json.loads(self.ledger.read_text())
        self.assertEqual(ledger["stage_used_gpu_seconds"]["reserve"], 9.)
        self.assertEqual(ledger["stage_used_gpu_seconds"]["direction1"], 0.)

    def test_cleanup_kills_remaining_group_after_leader_exited(self):
        child = Mock(pid=424242)
        child.poll.return_value = 0  # The leader is already reaped.
        killed, signals, elapsed = [False], [], [0.]
        def killpg(pgid, signum):
            self.assertEqual(pgid, child.pid)
            signals.append(signum)
            if killed[0]:
                raise ProcessLookupError()
            if signum == signal.SIGKILL:
                killed[0] = True
        def monotonic():
            elapsed[0] += 1.
            return elapsed[0]
        with patch.object(runner.os, "killpg", side_effect=killpg), \
             patch.object(runner.time, "monotonic", side_effect=monotonic), \
             patch.object(runner.time, "sleep"):
            self.assertTrue(runner.stop_owned_group(child))
        self.assertIn(signal.SIGTERM, signals)
        self.assertIn(signal.SIGKILL, signals)


if __name__ == "__main__":
    unittest.main()
