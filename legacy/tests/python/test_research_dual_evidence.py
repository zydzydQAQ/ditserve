import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ditserve.research.budget import BudgetStop
from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research.dual_evidence import (INVALIDATIONS_FILE, UnitJournal, invalidate_observation,
                                           invalidate_pending, read_records)
from ditserve.research.isolation import GPUIsolationError


class DualEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frozen = dict(sha256="protocol", source_sha256="source")
        self.snapshots = []
        def observe(*args, **kwargs):
            return dict(self.snapshots.pop(0)) if self.snapshots else dict(status="exclusive_observed", devices=["GPU-A"])
        self.journal = UnitJournal(self.root, self.frozen, ["GPU-A"], observe=observe)

    def unit(self, unit_id="unit"):
        return self.journal.unit(unit_id, 1., time.monotonic() + 20)

    def reseal_unit(self, unit):
        unit.pop("sha256", None)
        unit["sha256"] = digest(unit)
        atomic_json(self.root / "units" / (unit["unit_id"] + ".json"), unit)

    def test_completed_unit_requires_start_and_completed_observations_and_source(self):
        with self.unit():
            row = self.journal.append(dict(type="test", status="completed"))
        self.assertTrue(read_records(self.root, self.frozen)[0]["performance_valid"])
        unit = read_json(self.root / "units/unit.json")
        self.assertEqual([r["phase"] for r in unit["observations"]], ["start", "completed"])
        self.assertEqual(unit["records"], [row["sha256"]])
        for change in (dict(observations=unit["observations"][:1]), dict(source_sha256="other"),
                       dict(observations=[]), dict(performance_valid=False)):
            self.reseal_unit(dict(unit, **change))
            self.assertFalse(read_records(self.root, self.frozen)[0]["performance_valid"])

    def test_midunit_and_end_contention_invalidate_only_the_affected_unit(self):
        with self.unit("good"):
            self.journal.append(dict(case_id="good"))
        self.snapshots = [dict(status="exclusive_observed"), dict(status="contended", foreign=[dict(pid=98765)])]
        with self.assertRaises(GPUIsolationError):
            with self.unit("bad"):
                self.journal.append(dict(case_id="bad"))
        rows = read_records(self.root, self.frozen)
        self.assertEqual([r["performance_valid"] for r in rows], [True, False])
        self.assertIsNone(self.journal.active)
        self.snapshots = [dict(status="exclusive_observed"), dict(status="unknown")]
        with self.assertRaises(GPUIsolationError):
            with self.unit("poll-failure"):
                self.journal.append(dict(case_id="poll-failure"))
                self.journal.observation("poll")
        self.assertFalse(read_records(self.root, self.frozen)[-1]["performance_valid"])

    def test_explicit_invalid_measurement_cannot_be_rehabilitated_by_clean_unit(self):
        with self.unit():
            self.journal.append(dict(status="completed", performance_valid=False))
        self.assertFalse(read_records(self.root, self.frozen)[0]["performance_valid"])

    def test_failed_body_and_missing_seal_are_never_valid_cost_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with self.unit():
                self.journal.append(dict(status="completed"))
                raise RuntimeError("failure")
        self.assertFalse(read_records(self.root, self.frozen)[0]["performance_valid"])
        (self.root / "units/unit.json").unlink()
        self.assertFalse(read_records(self.root, self.frozen)[0]["performance_valid"])

    def test_pending_invalidation_preserves_a_completed_unit(self):
        with self.unit():
            self.journal.append(dict(status="completed"))
        before = (self.root / "units/unit.json").read_bytes()
        atomic_json(self.root / "active-unit.json", dict(status="running", unit_id="unit"))
        invalidate_pending(self.root, "parent stopped")
        self.assertEqual((self.root / "units/unit.json").read_bytes(), before)
        active = dict(status="running", unit_id="pending", protocol_sha256="protocol", source_sha256="source", records=[])
        atomic_json(self.root / "active-unit.json", active)
        invalidate_pending(self.root, "parent stopped")
        self.assertEqual(read_json(self.root / "units/pending.json")["status"], "failed")

    def test_nonfinite_budget_unsafe_ids_duplicate_units_and_append_outside_unit_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.journal.append(dict(status="completed"))
        for unit_id in ("../escape", "/absolute", "", "..", "bad\\id"):
            with self.assertRaises(ValueError):
                with self.unit(unit_id):
                    pass
        for estimate in (float("nan"), float("inf"), -1, 0, True):
            with self.assertRaises(ValueError):
                with self.journal.unit("invalid", estimate, time.monotonic() + 20):
                    pass
        with self.assertRaises(BudgetStop):
            with self.journal.unit("late", 10, time.monotonic() + 1):
                pass
        with self.unit():
            self.journal.append(dict(status="completed"))
        with self.assertRaises(ValueError):
            with self.unit():
                pass

    def test_observation_tampering_and_record_source_mismatch_fail_closed(self):
        with self.unit():
            self.journal.append(dict(status="completed"))
        unit = read_json(self.root / "units/unit.json")
        unit["observations"][-1]["status"] = "contended"
        self.reseal_unit(unit)
        self.assertFalse(read_records(self.root, self.frozen)[0]["performance_valid"])
        with self.assertRaisesRegex(ValueError, "different protocol/source"):
            read_records(self.root, dict(self.frozen, source_sha256="other"))

    def timed_unit(self, name, start, end):
        with patch("ditserve.research.dual_evidence.time.time", return_value=start) as clock:
            with self.unit(name):
                self.journal.append(dict(case_id=name, status="completed"))
                clock.return_value = end

    def test_completed_within_parent_query_is_invalidated_but_earlier_unit_is_preserved(self):
        self.timed_unit("earlier", 10., 19.)
        self.timed_unit("query-race", 19., 21.)
        self.timed_unit("later", 24., 28.)
        before = {p.name: p.read_bytes() for p in (self.root / "units").glob("*.json")}
        self.assertEqual(read_json(self.root / "active-unit.json")["status"], "idle")
        invalidate_observation(self.root, self.frozen, 20., 23., "foreign process returned after unit completed")
        rows = read_records(self.root, self.frozen)
        self.assertEqual([r["performance_valid"] for r in rows], [True, False, True])
        self.assertEqual({p.name: p.read_bytes() for p in (self.root / "units").glob("*.json")}, before)

    def test_parent_invalidation_written_while_active_still_applies_after_completed_seal(self):
        with patch("ditserve.research.dual_evidence.time.time", return_value=100.):
            with self.unit():
                self.journal.append(dict(status="completed"))
                invalidate_observation(self.root, self.frozen, 99., 101., "foreign process during unit")
        self.assertEqual(read_json(self.root / "units/unit.json")["status"], "completed")
        self.assertFalse(read_records(self.root, self.frozen)[0]["performance_valid"])

    def test_inclusive_boundaries_multiple_windows_and_invalid_timing_fail_closed(self):
        self.timed_unit("ends-at-start", 10., 20.)
        self.timed_unit("starts-at-end", 23., 25.)
        self.timed_unit("second-window", 40., 45.)
        invalidate_observation(self.root, self.frozen, 20., 23., "first")
        invalidate_observation(self.root, self.frozen, 42., 43., "second")
        self.assertEqual([r["performance_valid"] for r in read_records(self.root, self.frozen)], [False] * 3)
        self.timed_unit("unknown-time", 60., 61.)
        unit = read_json(self.root / "units/unknown-time.json")
        del unit["ended_at"]
        self.reseal_unit(unit)
        self.assertFalse(read_records(self.root, self.frozen)[-1]["performance_valid"])

    def test_parent_sidecar_rejects_tampering_foreign_source_and_invalid_intervals(self):
        self.timed_unit("unit", 10., 19.)
        for start, end in ((float("nan"), 20.), (20., float("inf")), (20., 19.), (True, 2.)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                invalidate_observation(self.root, self.frozen, start, end, "invalid")
        record = invalidate_observation(self.root, self.frozen, 20., 23., "valid")
        path = self.root / INVALIDATIONS_FILE
        bad = dict(record, ended_at=24.)
        path.write_text(json.dumps(bad) + "\n")
        with self.assertRaisesRegex(ValueError, "not bound"):
            read_records(self.root, self.frozen)
        bad = dict(record, source_sha256="foreign")
        bad.pop("sha256")
        bad["sha256"] = digest(bad)
        path.write_text(json.dumps(bad) + "\n")
        with self.assertRaisesRegex(ValueError, "not bound"):
            read_records(self.root, self.frozen)
        with self.assertRaisesRegex(ValueError, "not bound"):
            invalidate_observation(self.root, self.frozen, 30., 31., "must not append over foreign evidence")


if __name__ == "__main__":
    unittest.main()
