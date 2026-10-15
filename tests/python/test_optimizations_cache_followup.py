import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ditserve.optimizations import cache_followup as followup
from ditserve.research.common import atomic_json, digest
from ditserve.optimizations.budget import BudgetStop


class CacheFollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "archived/run01"
        self.source.mkdir(parents=True)
        self.out = self.root / "new/run01"
        self.ledger = self.out.parent / "gpu-budget.json"
        self.auth = self.root / "authorization.json"
        self.authorization(10800)

    def authorization(self, seconds, approved=True):
        atomic_json(self.auth, dict(approved_by_user=approved, authorized_gpu_seconds=seconds,
            user_instruction="Explicit fixture authorization", scope="cache_online_factorial_only"))

    def test_authorization_is_explicit_finite_and_scoped(self):
        for seconds, approved in [(10800, False), (0, True), (-1, True), (True, True)]:
            self.authorization(seconds, approved)
            with self.assertRaises(ValueError):
                followup.authorization(self.auth)
        self.authorization(10800)
        value, bound = followup.authorization(self.auth)
        self.assertEqual(value["authorized_gpu_seconds"], 10800)
        self.assertEqual(bound["path"], str(self.auth))

    def test_historical_paths_are_never_writable_supplement_destinations(self):
        for output, ledger in [(self.source / "new", self.ledger), (self.out, self.source / "gpu-budget.json"),
                              (followup.SOURCE_ROOT, self.ledger), (self.out, followup.OLD_LEDGER)]:
            with self.assertRaises(ValueError):
                followup._new_paths(output, ledger, self.source)

    def test_unlimited_requires_explicit_flag_and_null_limit(self):
        for flag, seconds in [(False, None), (True, 10800), (1, None), (False, float("inf"))]:
            self.auth.write_text(json.dumps(dict(approved_by_user=True, budget_unlimited=flag,
                authorized_gpu_seconds=seconds, user_instruction="Explicit fixture authorization",
                scope="cache_online_factorial_only")))
            with self.assertRaises(ValueError):
                followup.authorization(self.auth)
        atomic_json(self.auth, dict(approved_by_user=True, budget_unlimited=True,
            authorized_gpu_seconds=None, user_instruction="Explicit fixture authorization",
            scope="cache_online_factorial_only"))
        value, _ = followup.authorization(self.auth)
        self.assertTrue(value["budget_unlimited"])
        self.assertIsNone(value["authorized_gpu_seconds"])

    def test_missing_authorization_cli_stops_before_gpu_or_ledger(self):
        with patch.object(followup, "load_config", return_value={}), patch.object(followup, "execute") as execute:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                followup.main(["run"])
            self.assertEqual(error.exception.code, 2)
            execute.assert_not_called()
        self.assertFalse(self.ledger.exists())

    def test_insufficient_allowance_stops_before_resource_or_ledger(self):
        self.authorization(100)
        with patch.object(followup, "inspect_study", return_value=dict(estimate=dict(required_before_loading_seconds=200))):
            with self.assertRaises(BudgetStop):
                followup.execute({}, self.out, self.ledger, self.source, self.source / "supervisor.json", self.auth)
        self.assertFalse(self.ledger.exists())
        self.assertFalse(self.out.exists())

    def test_estimate_keeps_frozen_arrivals_and_all_sixteen_traces(self):
        traces = [dict(seed=102, load=.5, pattern="burst", trace_sha256="a"*64,
                       requests=[dict(arrival_ms=10000, service_ms=2000, kind="video")]),
                  dict(seed=101, load=.5, pattern="ordinary", trace_sha256="b"*64,
                       requests=[dict(arrival_ms=20000, service_ms=500, kind="image")])]
        original = json.dumps(traces, sort_keys=True)
        with patch("ditserve.optimizations.online.load_frozen_traces", return_value=traces):
            result = followup.estimate(dict(old_online="fixture"))
        self.assertEqual(result["traces"], 16)
        self.assertEqual(result["requests"], 384)
        self.assertEqual(result["minimum_arrival_window_seconds"], 240)
        self.assertAlmostEqual(result["online_admission_seconds"], (12+21)*8*1.08+60)
        self.assertEqual(json.dumps(traces, sort_keys=True), original)

    def execute_fixture(self, child_returncode=0, change_parent=False, unlimited=False):
        if unlimited:
            atomic_json(self.auth, dict(approved_by_user=True, budget_unlimited=True,
                authorized_gpu_seconds=None, user_instruction="Explicit fixture authorization",
                scope="cache_online_factorial_only"))
        old, research = self.root / "old/gpu-budget.json", self.root / "research/gpu-budget.json"
        for path in (old, research):
            atomic_json(path, dict(active=None, used_gpu_seconds=10., limit_gpu_seconds=100.))
        config = dict(gpu_uuid="GPU-fixture-A100", python="python", torch_home=str(self.root))
        source = dict(execution_root=str(self.root / "frozen"), source_sha256=digest({}), source_files={})
        prepared = dict(estimate=dict(required_before_loading_seconds=200), defaults=followup.DEFAULTS,
            replay=dict(path=str(self.source / "supervisor.json"), sha256="a"*64), old_ledgers=[])
        process = MagicMock(pid=12345)
        def waited(timeout):
            if change_parent:
                atomic_json(old, dict(active=None, used_gpu_seconds=11., limit_gpu_seconds=100.))
            return child_returncode
        process.wait.side_effect = waited
        def started(*args, **kwargs):
            self.assertEqual("--deadline" in args[0], not unlimited)
            atomic_json(self.out / "cache_k4/attempt00/worker.json",
                dict(status="passed", local_qualified=True, online_qualified=True, online_evaluation_complete=True))
            return process
        occupancy = dict(status="exclusive_observed", owned=[], foreign=[])
        with patch.object(followup, "OLD_LEDGER", old), patch.object(followup, "PARENT", research), \
             patch.object(followup, "inspect_study", return_value=prepared), \
             patch("ditserve.optimizations.runner.inspect", return_value=dict(status="ready")), \
             patch.object(followup, "freeze", return_value=source), \
             patch.object(followup, "gpu_occupancy", return_value=occupancy), \
             patch.object(followup, "stop_owned_group", return_value=True), \
             patch.object(followup.subprocess, "Popen", side_effect=started):
            return followup.execute(config, self.out, self.ledger, self.source,
                                    self.source / "supervisor.json", self.auth)

    def test_nonzero_worker_cannot_promote_candidate_and_cost_is_kept(self):
        result = self.execute_fixture(child_returncode=1)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["online_qualified"])
        self.assertGreaterEqual(result["charge"]["gpu_seconds"], 0)
        self.assertIsNone(json.loads(self.ledger.read_text())["active"])

    def test_parent_change_cannot_promote_successful_worker(self):
        result = self.execute_fixture(change_parent=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["online_qualified"])
        self.assertEqual(result["charge"]["status"], "failed")
        self.assertIn("parent_validation_error", result["charge"])

    def test_completed_online_attempt_settles_separate_ledger(self):
        result = self.execute_fixture()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["online_qualified"])
        self.assertEqual(result["charge"]["gpu_count"], 1)
        self.assertEqual(result["final_occupancy"]["owned"], [])

    def test_unlimited_execution_has_no_deadline_and_keeps_actual_cost(self):
        result = self.execute_fixture(unlimited=True)
        ledger = json.loads(self.ledger.read_text())
        self.assertEqual(result["status"], "passed")
        self.assertIsNone(ledger["limit_gpu_seconds"])
        self.assertTrue(ledger["budget_unlimited"])
        self.assertTrue(result["charge"]["identity"]["budget_unlimited"])
        self.assertGreaterEqual(result["charge"]["gpu_seconds"], 0)
        self.assertEqual(ledger["used_gpu_seconds"], result["charge"]["gpu_seconds"])

    def test_unlimited_guard_still_stops_when_supervisor_exits(self):
        guard = followup.BudgetGuard(float("inf"), supervisor_pid=1234)
        with patch("os.getppid", return_value=1234):
            self.assertEqual(guard.remaining(), float("inf"))
            guard.ensure_unit(1e9)
        with patch("os.getppid", return_value=1235), self.assertRaises(BudgetStop):
            guard()


if __name__ == "__main__":
    unittest.main()
