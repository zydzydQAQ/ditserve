import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.directions.budget import BudgetStop
from ditserve.directions.worker import main, verify_admission
from ditserve.research.common import digest


class WorkerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.args = SimpleNamespace(ledger=self.root / "budget.json", session_id="charged-session",
                                    output=self.root / "direction2", direction=2, phase="screen")
        self.config = dict(device="cuda:0", gpu_uuid="GPU-selected")
        self.active = dict(session_id=self.args.session_id, supervisor_pid=101, child_pgid=202,
                           stage="direction2", gpu_count=1, identity=dict(direction=2, phase="screen",
                           output=str(self.args.output.resolve()), config_sha256=digest(self.config)))
        self.clock = 100.
        self.write()

    def tearDown(self):
        self.directory.cleanup()

    def write(self):
        self.args.ledger.write_text(json.dumps(dict(active=self.active)))

    def verify(self, **kwargs):
        defaults = dict(getpid=lambda: 202, getpgrp=lambda: 202, getppid=lambda: 101,
                        environ={"CUDA_VISIBLE_DEVICES": "GPU-selected"},
                        monotonic=lambda: self.clock, sleep=self.advance)
        return verify_admission(self.args, self.config, **dict(defaults, **kwargs))

    def advance(self, seconds):
        self.clock += seconds

    def test_registered_session_admits_only_its_child(self):
        self.assertEqual(self.verify()["session_id"], self.args.session_id)
        for overrides in ({"getppid": lambda: 1}, {"getpid": lambda: 303},
                          {"getpgrp": lambda: 303}, {"environ": {}}):
            with self.subTest(overrides=overrides), self.assertRaises(BudgetStop):
                self.verify(**overrides)

    def test_all_scientific_bindings_are_required(self):
        original = copy.deepcopy(self.active)
        for field, value in (("session_id", "other"), ("stage", "direction1"), ("gpu_count", 2),
                             ("child_pgid", 404), ("supervisor_pid", 909)):
            self.active = copy.deepcopy(original)
            self.active[field] = value
            self.write()
            with self.subTest(field=field), self.assertRaises(BudgetStop):
                self.verify()
        for field, value in (("direction", 1), ("phase", "qualify"),
                             ("output", "/other"), ("config_sha256", "different")):
            self.active = copy.deepcopy(original)
            self.active["identity"][field] = value
            self.write()
            with self.subTest(field=field), self.assertRaises(BudgetStop):
                self.verify()

    def test_child_registration_race_is_bounded_and_can_complete(self):
        self.active["child_pgid"] = None
        self.write()
        def register(seconds):
            self.advance(seconds)
            self.active["child_pgid"] = 202
            self.write()
        self.assertEqual(self.verify(sleep=register)["child_pgid"], 202)
        self.active["child_pgid"] = None
        self.write()
        with self.assertRaisesRegex(BudgetStop, "did not register"):
            self.verify()
        self.assertLess(self.clock, 105.2)

    def test_missing_or_settled_allocation_cannot_admit(self):
        self.active = None
        self.write()
        with self.assertRaises(BudgetStop):
            self.verify()
        self.args.ledger.unlink()
        with self.assertRaises(BudgetStop):
            self.verify()

    def test_rejected_admission_never_imports_direction_or_model(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        for phase in ("screen", "state_cost"):
            with self.subTest(phase=phase), \
                 patch("ditserve.directions.worker.verify_admission", side_effect=BudgetStop("denied")), \
                 patch("ditserve.directions.worker.importlib.import_module") as import_module:
                result = main(["--config", str(config_path), "--output", str(self.args.output),
                               "--direction", "2", "--phase", phase, "--deadline", "999999999",
                               "--ledger", str(self.args.ledger), "--session-id", self.args.session_id])
            self.assertEqual(result, 2)
            import_module.assert_not_called()
            self.assertFalse((self.args.output / ("environment-%s.json" % phase)).exists())
            report = json.loads((self.args.output / ("worker-%s.json" % phase)).read_text())
            self.assertEqual(report["status"], "incomplete")

    def reserve(self, **overrides):
        path = self.root / "supervisor-screen.json"
        failed = dict(status="failed", charge=dict(gpu_seconds=10.), direction=2,
                      phase="screen", config_sha256=digest(self.config))
        failed.update(overrides)
        path.write_text(json.dumps(failed))
        self.active["stage"] = "reserve"
        self.active["identity"]["reserve_for"] = dict(path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), direction=2, phase="screen")
        self.write()
        return path

    def test_reserve_requires_charged_failed_same_phase_and_protocol(self):
        for status in ("failed", "incomplete"):
            self.reserve(status=status)
            self.assertEqual(self.verify()["stage"], "reserve")
        for invalid in (dict(status="passed"), dict(status="stopped"), dict(charge={}),
                        dict(direction=1), dict(phase="qualify"), dict(config_sha256="changed")):
            self.reserve(**invalid)
            with self.subTest(invalid=invalid), self.assertRaises(BudgetStop):
                self.verify()

    def test_reserve_rechecks_original_evidence_and_binding(self):
        path = self.reserve()
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(BudgetStop):
            self.verify()
        path.unlink()
        with self.assertRaises(BudgetStop):
            self.verify()
        for invalid in (None, dict(path="relative.json", direction=2, phase="screen"),
                        dict(path=str(path), direction=3, phase="screen")):
            self.active["identity"]["reserve_for"] = invalid
            self.write()
            with self.subTest(invalid=invalid), self.assertRaises(BudgetStop):
                self.verify()


if __name__ == "__main__":
    unittest.main()
