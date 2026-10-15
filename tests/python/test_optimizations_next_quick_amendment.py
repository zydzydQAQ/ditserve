import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.optimizations import next_quick_amendment as module
from ditserve.research.common import atomic_json, digest, file_digest, read_json


class QuickAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run01"
        self.snapshot = self.root / "snapshot"
        self.args = SimpleNamespace(phase="cache_load_09", run_root=self.run,
            ledger=self.root / "gpu-budget.json", output=self.run / "cache_load_09/attempt00")
        self.config = dict(fixture=True)
        auth_path = self.root / "auth.json"
        atomic_json(auth_path, dict(approved_by_user=True))
        self.auth = dict(path=str(auth_path), sha256=file_digest(auth_path))
        self.note = dict(schema_version="ditserve.optimizations.quick_amendment.v1",
            approved_by_user=True, user_instruction=module.INSTRUCTION, phase="cache_load_09",
            load=.9, seed=101, patterns=["ordinary", "burst"],
            orders=[["baseline", "candidate"], ["candidate", "baseline"]],
            planned_replays=8, planned_requests=192, full_qualification=False,
            run_root=str(self.run), ledger=str(self.args.ledger), base_config_sha256=digest(self.config),
            workloads_sha256="workloads", authorization_binding=self.auth, after_session_id="previous07")
        self.active = dict(stage="cache_load_09", session_id="next09", identity=dict(
            config_sha256=digest(self.config), workloads_sha256="workloads",
            authorization_source=self.auth, source_sha256="snapshot-source"))
        self.charge = dict(session_id="previous07", stage="cache_load_07", status="completed")
        self.prior = dict(status="completed", returncode=0, recorded_rows=24, evaluation_complete=True,
            charge=self.charge, config_sha256=digest(self.config),
            final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[]))
        self.prior_path = self.run / "cache_load_07/terminal.json"
        atomic_json(self.prior_path, self.prior)
        atomic_json(self.args.ledger, dict(active=self.active, sessions=[self.charge]))
        self.write_note()

    def write_note(self):
        path = self.snapshot / module.RELATIVE_PATH
        atomic_json(path, self.note)
        atomic_json(self.snapshot / "manifest.json", dict(source_sha256="snapshot-source",
            source_files={module.RELATIVE_PATH: file_digest(path)}))

    def admit(self):
        return module.admit_quick_amendment(self.args, self.config, self.active, self.snapshot)

    def test_only_admits_after_24_replays_and_preserves_prior_files(self):
        before = {p: p.read_bytes() for p in (self.prior_path, self.args.ledger)}
        result = self.admit()
        self.assertEqual(result["planned_replays"], 8)
        self.assertEqual(result["seed"], 101)
        self.assertFalse(result["full_qualification"])
        self.assertEqual(read_json(self.args.output / "quick-amendment-admission.json"), result)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_no_amendment_applied_to_running_07_or_other_phase(self):
        self.args.phase = "cache_load_07"
        self.assertIsNone(self.admit())
        self.assertFalse(self.args.output.exists())

    def test_requires_snapshot_bound_scientific_scope(self):
        original = copy.deepcopy(self.note)
        for field, value in (("seed", 102), ("planned_replays", 24), ("full_qualification", True),
                             ("user_instruction", "not authorized"), ("base_config_sha256", "wrong")):
            self.note = dict(original, **{field: value})
            self.write_note()
            with self.assertRaises(ValueError):
                self.admit()
        self.note = original
        self.write_note()
        path = self.snapshot / module.RELATIVE_PATH
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.admit()

    def test_rejects_incomplete_failed_or_unclean_07(self):
        for changes in (dict(recorded_rows=23), dict(evaluation_complete=False),
                        dict(returncode=1), dict(status="stopped"),
                        dict(final_occupancy=dict(status="exclusive_observed", owned=[123], foreign=[]))):
            atomic_json(self.prior_path, dict(self.prior, **changes))
            with self.assertRaises(ValueError):
                self.admit()

    def test_rejects_other_or_unsettled_session(self):
        for budget in (dict(active=self.active, sessions=[]),
                       dict(active=dict(session_id="other"), sessions=[self.charge]),
                       dict(active=None, sessions=[self.charge])):
            atomic_json(self.args.ledger, budget)
            with self.assertRaises(ValueError):
                self.admit()

    def test_context_hook_reads_only_named_future_phase(self):
        context = SimpleNamespace(run_root=self.run, ledger=self.args.ledger,
            output=self.args.output, config=self.config, metadata={})
        with patch("ditserve.optimizations.protocol.ROOT", self.snapshot):
            self.assertIsNone(module.admit_context_amendment(context, "cache_load_07"))
            result = module.admit_context_amendment(context, "cache_load_09")
        self.assertEqual(context.metadata["quick_amendment"], result)

    def test_next_worker_uses_amendment_without_explicit_quick_flag(self):
        from test_optimizations_next_workloads import Fixture, measured
        from ditserve.optimizations import next_workloads as workloads
        fixture_root = self.root / "fixture"
        fixture_root.mkdir()
        fixture = Fixture(fixture_root)
        traces = fixture.build()["highload"]["0.9"]
        self.config = fixture.config
        self.note["base_config_sha256"] = digest(self.config)
        self.active["identity"]["config_sha256"] = digest(self.config)
        self.prior["config_sha256"] = digest(self.config)
        atomic_json(self.prior_path, self.prior)
        atomic_json(self.args.ledger, dict(active=self.active, sessions=[self.charge]))
        self.write_note()
        context = fixture.context()
        context.output, context.run_root, context.ledger = self.args.output, self.run, self.args.ledger
        with patch("ditserve.optimizations.protocol.ROOT", self.snapshot), \
                patch.object(workloads, "run_trace", side_effect=measured) as replay:
            result = workloads.run_cache_load(context, traces, "cache_load_09")
        self.assertEqual(replay.call_count, 8)
        self.assertEqual(result["status"], "stopped")
        self.assertTrue(result["screening_complete"])
        self.assertFalse(result["online_qualified"])
        self.assertFalse(result["evaluation_complete"])
        self.assertEqual({row["seed"] for row in result["rows"]}, {101})
        report = read_json(self.args.output / "quick-report.json")
        self.assertEqual(report["recorded_replays"], 8)
        self.assertEqual(report["excel"]["status"], "verified")


if __name__ == "__main__":
    unittest.main()
