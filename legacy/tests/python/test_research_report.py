import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ditserve.research.budget import ResearchBudget
from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.research.report import (build_report, make_figures, quality_cost_points,
                                     read_measurement_environment, read_records)


def environment_record(frozen, status="exclusive_observed", valid=True, root=None):
    stage = dict(status=status, observations_count=2, log="mechanism-gpu-isolation.jsonl",
                 completion_observed=True, worker_returncode=0)
    if root is not None:
        path = Path(root) / stage["log"]
        path.write_text('fixture start and successful completion\n')
        stage["log_sha256"] = file_digest(path)
    record = dict(schema_version="ditserve.research.measurement_environment.v1",
                  protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"],
                  status=status, cost_measurements_valid=valid,
                  stages={"mechanism": stage})
    return dict(record, sha256=digest(record))


def quality_rows():
    # Deliberately distinct cost fields make accidental mixing observable.
    return [dict(type="calibration", status="completed", split="dev", case_id="initial",
                 phase_cost_ms=dict(prepare=90000, denoise=2000, decode=1000), cost_ms=99000,
                 evidence=dict(metrics=dict(lpips_mean=.02))),
            dict(type="joint_quality", status="unqualified", split="holdout", case_id="joint",
                 phase_cost_ms=dict(denoise=10, decode=20), cost_ms=7000, cost_scope="full_critical_path",
                 evidence=dict(metrics=dict(lpips_mean=.07)))]


class ResearchReportTests(unittest.TestCase):
    def test_missing_measurements_never_become_negative_or_positive_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "budget.json"
            ResearchBudget(ledger).initialize()
            frozen = dict(ledger=str(ledger), sha256="a" * 64, source_sha256="b" * 64)
            atomic_json(root / "preflight.json", dict(status="blocked", reasons=["driver unavailable"]))
            with patch("ditserve.research.report.make_figures", return_value=dict(status="completed", paths=[])):
                report = build_report(root, frozen)
            self.assertEqual(report["measured_records"], 0)
            self.assertEqual(report["gpu_seconds_used"], 0)
            for name in ("memory_admission", "state_crossover", "sp_path"):
                self.assertEqual(report["gates"][name]["status"], "insufficient_evidence")
            self.assertTrue((root / "REPORT.md").is_file())
            saved = read_json(root / "report.json")
            self.assertEqual(saved["measurement_environment"]["status"], "legacy_unmonitored")
            self.assertFalse(saved["cost_measurements_valid"])
            self.assertEqual(saved["frozen_source_sha256"], frozen["source_sha256"])
            self.assertEqual(saved["report_generator_source_sha256"], digest(saved["report_generator_source_files"]))
            self.assertFalse(saved["report_generator_matches_frozen_source"])

    def test_contended_environment_blocks_raw_supported_gates_but_retains_correctness(self):
        import json
        from test_research_evidence import pair
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "budget.json"
            ResearchBudget(ledger).initialize()
            frozen = dict(ledger=str(ledger), sha256="a" * 64, source_sha256="b" * 64)
            rows = [pair("small", repeat, dict(keep=90., hot=100.)) for repeat in (0, 1)]
            rows += [pair("large", repeat, dict(keep=100., hot=90.)) for repeat in (0, 1)]
            rows = [dict(row, type="state_trial") for row in rows]
            rows.append(dict(type="correctness", case_id="official-off", status="passed", official_equal=True))
            records = []
            for row in rows:
                row.update(protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"])
                records.append(dict(row, sha256=digest(row)))
            raw = "".join(json.dumps(row) + "\n" for row in records)
            (root / "records.jsonl").write_text(raw)
            for status, valid, expected in (("exclusive_observed", True, "supported"),
                                             ("contended", False, "insufficient_evidence"),
                                             ("unknown", False, "insufficient_evidence")):
                atomic_json(root / "measurement-environment.json", environment_record(frozen, status, valid, root))
                with patch("ditserve.research.report.make_figures", return_value=dict(status="completed", paths=[])):
                    result = build_report(root, frozen)
                self.assertEqual(result["gates"]["state_crossover"]["status"], expected)
                saved = read_json(root / "report.json")
                self.assertEqual(saved["numerical_correctness"], [dict(case_id="official-off", status="passed", official_equal=True)])
                self.assertEqual((root / "records.jsonl").read_text(), raw)
                self.assertEqual(saved["measured_records"], 5)

    def test_missing_invalid_or_foreign_environment_never_allows_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = dict(sha256="a" * 64, source_sha256="b" * 64)
            self.assertEqual(read_measurement_environment(root, frozen)["status"], "legacy_unmonitored")
            for field, value in (("protocol_sha256", "c" * 64), ("source_sha256", "d" * 64),
                                 ("schema_version", "unrelated"), ("cost_measurements_valid", "true")):
                record = environment_record(frozen)
                record[field] = value
                record.pop("sha256")
                record["sha256"] = digest(record)
                atomic_json(root / "measurement-environment.json", record)
                observed = read_measurement_environment(root, frozen)
                self.assertEqual(observed["status"], "invalid")
                self.assertFalse(observed["cost_measurements_valid"])
            changed = environment_record(frozen)
            changed["stages"]["mechanism"]["observations_count"] = 99
            atomic_json(root / "measurement-environment.json", changed)
            self.assertEqual(read_measurement_environment(root, frozen)["status"], "invalid")
            (root / "measurement-environment.json").write_text("{interrupted")
            self.assertFalse(read_measurement_environment(root, frozen)["cost_measurements_valid"])

    def test_signed_invalid_cost_flag_and_inconsistent_exclusivity_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = dict(sha256="a" * 64, source_sha256="b" * 64)
            for status, valid in (("exclusive_observed", False), ("contended", True), ("unknown", True)):
                atomic_json(root / "measurement-environment.json", environment_record(frozen, status, valid))
                self.assertFalse(read_measurement_environment(root, frozen)["cost_measurements_valid"])

    def test_partial_or_changed_observation_log_cannot_qualify_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = dict(sha256="protocol", source_sha256="source")
            record = environment_record(frozen, root=root)
            atomic_json(root / "measurement-environment.json", record)
            self.assertTrue(read_measurement_environment(root, frozen)["cost_measurements_valid"])
            (root / "mechanism-gpu-isolation.jsonl").write_text("truncated")
            self.assertFalse(read_measurement_environment(root, frozen)["cost_measurements_valid"])
            record = environment_record(frozen, root=root)
            record["stages"]["mechanism"].update(completion_observed=False)
            record.pop("sha256")
            record["sha256"] = digest(record)
            atomic_json(root / "measurement-environment.json", record)
            self.assertFalse(read_measurement_environment(root, frozen)["cost_measurements_valid"])

    def test_quality_costs_are_separate_and_missing_scope_never_falls_back(self):
        rows = quality_rows()
        points = quality_cost_points(rows)
        self.assertEqual([p["cost_seconds"] for p in points["calibration"]], [3.])
        self.assertEqual([p["cost_seconds"] for p in points["joint"]], [7.])
        rows[1].pop("cost_scope")
        self.assertEqual(quality_cost_points(rows)["joint"], [])

    def test_figures_separate_scopes_and_hide_contended_performance(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("CPU matplotlib is required for figure rendering")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = make_figures(root, quality_rows(), environment=dict(status="exclusive_observed", cost_measurements_valid=True))
            panels = {row["path"]: row for row in result["paths"] if "quality-time" in row["path"]}
            self.assertEqual(set(panels), {"figures/quality-time-calibration.svg", "figures/quality-time-joint.svg"})
            self.assertEqual({p["timing_scope"] for p in panels.values()}, {"denoise_plus_decode", "full_critical_path"})
            self.assertTrue(all(p["measured"] for p in panels.values()))
            result = make_figures(root, quality_rows(), environment=dict(status="contended", cost_measurements_valid=False))
            self.assertFalse(any(row["measured"] for row in result["paths"]))
            for panel in result["paths"]:
                self.assertIn("No usable exclusive performance measurements", (root / panel["path"]).read_text())

    def test_modified_record_rejected_before_reporting(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = dict(sha256="a" * 64, source_sha256="b" * 64)
            row = dict(type="probe", status="completed", protocol_sha256=frozen["sha256"],
                       source_sha256=frozen["source_sha256"])
            row["sha256"] = digest(row)
            row["status"] = "failed"
            (root / "records.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid research evidence"):
                read_records(root, frozen)


if __name__ == "__main__":
    unittest.main()
