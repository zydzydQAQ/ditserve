import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ditserve.research import migration_diagnostic_cost as cost
from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research.dual_evidence import UnitJournal
from ditserve.research.dual_matrix import seal
from ditserve.research.migration_diagnostic_protocol import DEFAULT_CONFIG, FROZEN_SCHEMA


class MigrationDiagnosticCostTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent_root = self.root / "run03"
        self.math_root = self.root / "old-run02"
        self.config = dict(model_id="Wan1.3B", model_path="/fixed/model", revision="fixed-revision",
            precisions=dict(transformer="float16", vae="float32", text_encoder="float32"),
            steps=32, guidance_scale=5., shapes=dict(main=dict(cost.SHAPE), fallback=dict(cost.SHAPE)),
            recipes=[dict(id="off", cache=False, threshold=0.)])
        self.model = dict(sha256="fixed-model-content")
        self.parent = seal(dict(config=copy.deepcopy(self.config), model_manifest=self.model, source_sha256="parent-source"))
        self.old = seal(dict(config=copy.deepcopy(self.config), model_manifest=self.model, source_sha256="old-source"))
        self.journals = {}
        for root, frozen in ((self.parent_root, self.parent), (self.math_root, self.old)):
            atomic_json(root / "protocol.json", frozen)
            self.journals[root] = UnitJournal(root, frozen, ["A", "V"],
                observe=lambda *args, **kwargs: dict(status="exclusive_observed"))
        for role, backend, base in (("a100", "flash", 5.), ("v100", "efficient", 13.)):
            for index in range(20):
                self.add(self.parent_root, "formal-" + role + "-" + str(index), 30., dict(
                    type="cfg_calibration", case_id=role + "-" + str(index), device_role=role,
                    context_sha256="context", phase="dev" if index < 16 else "holdout",
                    status="completed", fresh_pair=True, timing_eligible=True,
                    reference=self.trajectory(role, backend, base + index / 1000.)))
        self.add(self.parent_root, "migration-reference", 15., dict(type="cfg_migration_reference",
            case_id="dev-kettle-102", status="failed"))
        self.session = dict(session_id="parent-migration", status="completed", gpu_count=2,
            stage="migration", protocol_sha256=self.parent["sha256"], wall_seconds=49., gpu_seconds=98.)
        anchor = seal(dict(parent_run=str(self.parent_root), parent_protocol_sha256=self.parent["sha256"],
            parent_source_sha256=self.parent["source_sha256"], parent_session_id=self.session["session_id"],
            parent_session_sha256=digest(self.session)))
        self.frozen = seal(dict(schema_version=FROZEN_SCHEMA, diagnostic=read_json(DEFAULT_CONFIG),
            source_sha256="new-source", config=copy.deepcopy(self.config), model_manifest=self.model,
            parent_failure=anchor, budget_binding=dict(stage="migration", limit_gpu_seconds=28800.,
                stage_limit_gpu_seconds=5500., reallocation_allowed=False, borrowing_allowed=False)))
        a = self.add(self.math_root, "math-a", 40., dict(type="backend_control", case_id="fallback",
            context_sha256="context", status="completed", a100_math=self.trajectory("a100", "math", 12.)))
        v = self.add(self.math_root, "math-v", 40., dict(type="probe", case_id="v100-fallback",
            context_sha256="context", status="completed", run=self.trajectory("v100", "math", 24.)))
        proxy = self.add(self.math_root, "math-pair", 38., dict(type="backend_pair", case_id="fallback-0",
            context_sha256="context", status="completed", shape=dict(cost.SHAPE),
            runs=dict(math=self.trajectory("v100", "math", 23.), efficient=self.trajectory("v100", "efficient", 13.))))
        self.pins = dict(a100=a["sha256"], v100=v["sha256"], evaluation_proxy=proxy["sha256"])
        for name, value in (("MATH_ROOT", self.math_root), ("MATH_PROTOCOL_SHA256", self.old["sha256"]),
                            ("MATH_RECORD_SHA256", self.pins)):
            mocked = patch.object(cost, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.ledger = dict(active=None, limit_gpu_seconds=28800., used_gpu_seconds=15000.,
            stage_limits_gpu_seconds=dict(migration=5500., correctness=100., reserve=2000.),
            stage_used_gpu_seconds=dict(migration=100., correctness=10., reserve=100.),
            sessions=[self.session])

    @staticmethod
    def trajectory(role, backend, seconds):
        return dict(status="completed", recipe_id="off", device_role=role, all_requests_decoded=True,
            elapsed_ms=seconds * 1000, identity=dict(recipe="off", path=[role], action="stay",
                attention_path=[backend], context_sha256="context"),
            events=[dict(kind="step", step_index=i, completed_steps=i + 1, finite=True,
                path="compute", reused_forwards=0, threshold=0.) for i in range(32)] +
                [dict(kind="decode", finite=True, shape=[1, 17, 256, 448, 3])])

    def add(self, root, unit_id, seconds, value):
        journal = self.journals[root]
        with journal.unit(unit_id, 1., time.monotonic() + 60.):
            row = journal.append(value)
        path = root / "units" / (unit_id + ".json")
        unit = read_json(path)
        unit.pop("sha256")
        unit.update(started_at=100., ended_at=100. + seconds)
        atomic_json(path, seal(unit))
        return row

    def change_row(self, root, unit_id, mutate):
        path = root / "records.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        row = next(row for row in rows if row["unit_id"] == unit_id)
        before = row.pop("sha256")
        mutate(row)
        row.update(sha256=digest(row))
        path.write_text("".join(json.dumps(item) + "\n" for item in rows))
        unit_path = root / "units" / (unit_id + ".json")
        unit = read_json(unit_path)
        unit.pop("sha256")
        unit["records"] = [row["sha256"] if value == before else value for value in unit["records"]]
        atomic_json(unit_path, seal(unit))
        for name in self.pins:
            if self.pins[name] == before:
                self.pins[name] = row["sha256"]

    def test_complete_bound_counts_holding_cleanup_and_eight_evaluations(self):
        before_ledger, before_frozen = copy.deepcopy(self.ledger), copy.deepcopy(self.frozen)
        report = cost.cost_plan(self.frozen, self.ledger)
        self.assertEqual(report["status"], "fits")
        self.assertEqual(report["sha256"], digest({k: v for k, v in report.items() if k != "sha256"}))
        expected = {}
        for name, a, v in (("original", 5.019, 13.019), ("math", 12., 24.)):
            unit = (4 * a + 5 * v + 60 + 8 * 2) * 1.3
            phase = 35 * 1.3 + 4 * unit
            expected[name] = 2 * (phase + 30)
            value = report["comparisons"][name]
            self.assertAlmostEqual(value["unit_wall_seconds"], unit)
            self.assertAlmostEqual(value["phase_startup_wall_seconds"], phase)
            self.assertAlmostEqual(value["total_gpu_seconds"], expected[name])
            self.assertEqual(value["gpu_count"], 2)
            self.assertEqual(value["complete_units"], 4)
        self.assertAlmostEqual(report["total_gpu_seconds"], sum(expected.values()))
        self.assertEqual(report["remaining_stage_gpu_seconds"]["migration"], 5400.)
        self.assertEqual(report["remaining_stage_gpu_seconds"]["correctness"], 90.)
        self.assertEqual(report["borrowing_gpu_seconds"], 0.)
        self.assertEqual(report["optimization_savings_assumed_seconds"], 0.)
        self.assertEqual(self.ledger, before_ledger)
        self.assertEqual(self.frozen, before_frozen)
        for name in ("quality_qualified", "performance_qualified", "inherit_formal_evidence"):
            self.assertIs(report[name], False)

    def test_cost_evidence_binds_sealed_record_unit_protocol_and_maximum(self):
        report = cost.cost_plan(self.frozen, self.ledger)
        evidence = report["evidence"]
        for role in ("a100", "v100"):
            ref = evidence["original"][role]
            self.assertEqual(ref["protocol_sha256"], self.parent["sha256"])
            self.assertEqual(ref["case_id"], role + "-19")
            self.assertEqual(len(ref["considered_record_sha256"]), 20)
            self.assertEqual(ref["unit_sha256"], read_json(self.parent_root / "units" / (ref["unit_id"] + ".json"))["sha256"])
            self.assertEqual(evidence["math"][role]["record_sha256"], self.pins[role])
        self.assertEqual(evidence["math"]["evaluation_proxy"]["observed_residual_wall_seconds"], 2.)
        self.assertIn("not isolated LPIPS", evidence["math"]["evaluation_proxy"]["kind"])
        self.assertEqual(evidence["original"]["setup"]["measured_non_unit_wall_seconds"], 34.)

    def test_stage_and_lifetime_admission_are_independent(self):
        self.ledger["stage_used_gpu_seconds"]["migration"] = 5000.
        self.assertEqual(cost.cost_plan(self.frozen, self.ledger)["status"], "budget_stopped")
        self.ledger["stage_used_gpu_seconds"]["migration"] = 100.
        self.ledger["used_gpu_seconds"] = 28000.
        self.assertEqual(cost.cost_plan(self.frozen, self.ledger)["status"], "budget_stopped")

    def test_active_or_changed_cap_cannot_be_planned(self):
        self.ledger["active"] = dict(session_id="active")
        with self.assertRaisesRegex(ValueError, "settled"):
            cost.cost_plan(self.frozen, self.ledger)
        self.ledger["active"] = None
        self.ledger["stage_limits_gpu_seconds"]["migration"] += 1
        with self.assertRaisesRegex(ValueError, "caps"):
            cost.cost_plan(self.frozen, self.ledger)

    def test_invalidated_largest_measurement_is_not_silently_skipped(self):
        path = self.parent_root / "units/formal-v100-19.json"
        value = read_json(path)
        value.pop("sha256")
        value["performance_valid"] = False
        atomic_json(path, seal(value))
        with self.assertRaisesRegex(ValueError, "invalidated"):
            cost.cost_plan(self.frozen, self.ledger)

    def test_missing_formal_reference_cannot_reduce_maximum(self):
        path = self.parent_root / "records.jsonl"
        path.write_text("\n".join(line for line in path.read_text().splitlines()
            if json.loads(line)["unit_id"] != "formal-a100-19") + "\n")
        with self.assertRaisesRegex(ValueError, "all 16"):
            cost.cost_plan(self.frozen, self.ledger)

    def test_wrong_backend_shape_cache_or_incomplete_steps_reject(self):
        mutations = [lambda run: run["identity"].update(attention_path=["flash"]),
            lambda run: run["events"][-1].update(shape=[1, 17, 480, 832, 3]),
            lambda run: run.update(recipe_id="paired_008"),
            lambda run: run["events"][0].update(reused_forwards=1),
            lambda run: run["events"].pop(0)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                original = self.trajectory("v100", "math", 24.)
                self.change_row(self.math_root, "math-v", lambda row: row.update(run=copy.deepcopy(original)))
                self.change_row(self.math_root, "math-v", lambda row: mutate(row["run"]))
                with self.assertRaisesRegex(ValueError, "trajectory|steps"):
                    cost.cost_plan(self.frozen, self.ledger)

    def test_pinned_history_cannot_be_replaced(self):
        self.pins["a100"] = "different-sealed-row"
        with self.assertRaisesRegex(ValueError, "pinned"):
            cost.cost_plan(self.frozen, self.ledger)

    def test_tampered_journal_cannot_supply_cost(self):
        path = self.math_root / "records.jsonl"
        path.write_text(path.read_text().replace('"elapsed_ms": 24000.0', '"elapsed_ms": 1.0'))
        with self.assertRaisesRegex(ValueError, "digest"):
            cost.cost_plan(self.frozen, self.ledger)

    def test_model_and_cfg_specifications_must_match(self):
        bad = copy.deepcopy(self.frozen)
        bad.pop("sha256")
        bad["config"]["guidance_scale"] = 4.
        with self.assertRaisesRegex(ValueError, "specification"):
            cost.cost_plan(seal(bad), self.ledger)

    def test_negative_or_nonfinite_measurements_reject(self):
        self.change_row(self.math_root, "math-v", lambda row: row["run"].update(elapsed_ms=-1))
        with self.assertRaisesRegex(ValueError, "finite"):
            cost.cost_plan(self.frozen, self.ledger)

    def test_session_provenance_cannot_change_after_anchor(self):
        self.ledger["sessions"][0]["wall_seconds"] = 50.
        with self.assertRaisesRegex(ValueError, "settled bound"):
            cost.cost_plan(self.frozen, self.ledger)


if __name__ == "__main__":
    unittest.main()
