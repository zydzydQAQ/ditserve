import copy
import time
import unittest
from unittest.mock import patch

from ditserve.research import cfg_protocol
from ditserve.research.cfg_cost_plan import continuation_cost_plan, pending_cost_plan, _completed
from ditserve.research.cfg_matrix import select_candidate
from ditserve.research.cfg_refinement import (build_refinement, verify_refinement, development_candidates,
                                             correctness_complete, verify_cpu_binding)
from ditserve.research.common import atomic_json, read_json
from ditserve.research.dual_evidence import UnitJournal, read_records
from ditserve.research.dual_matrix import seal
from test_research_cfg_continuation import CfgContinuationTests
from test_research_cfg_matrix import cache_rows


class RefinementTests(unittest.TestCase):
    def setUp(self):
        fixture = CfgContinuationTests("test_parent_gate_only_allows_fresh_current_development_qualification")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fx, self.parent = fixture, fixture.new_freeze()
        self.root = fixture.root / "results/run02"
        self.journal = UnitJournal(self.root, self.parent, ["GPU-a100", "GPU-v100"],
                                   observe=lambda *a, **kw: dict(status="exclusive_observed"))
        self.ledger = dict(active=None, used_gpu_seconds=13000, limit_gpu_seconds=28800, sessions=[],
            stage_used_gpu_seconds={}, reallocations=[dict(baseline_stage_used={},
                stage_allowances_gpu_seconds=dict(correctness=500, calibration=2000, capacity=900))])
        for role in ("a100", "v100"):
            for recipe in ("off", "independent_008", "paired_008"):
                run = dict(elapsed_ms=9000., identity=dict(implementation="split_prefix"))
                records = [dict(type="cfg_correctness", recipe_id=recipe, equal=True,
                                continuous=run, restored=run)]
                if recipe == "paired_008":
                    records.append(dict(type="cfg_implementation_equivalence", recipe_id=recipe, equal=True))
                self.unit("correctness-" + role + "-" + recipe, records, role, 20.)
            self.unit("correctness-" + role + "-diagnostic", [dict(type="cfg_diagnostic_neutrality", equal=True,
                instrumented=dict(events=[dict(kind="step", cfg_research=dict(diagnostic_ms=32.)) for _ in range(32)]))], role, 20.)
            self.closed("correctness-" + role, dict(checks_passed=True), wall=85.)
            for row in fixture.current_development():
                if row["device_role"] != role:
                    continue
                row["candidate"]["elapsed_ms"] = 100.
                self.unit("calibrate-" + role + "-" + row["case_id"], [row], role, 10.)
            self.closed("calibrate-" + role, dict(candidates=["independent_004", "paired_008"]), wall=325.)
        self.rows = read_records(self.root, self.parent)
        selection = select_candidate(self.rows, fixture.config, "new-context", frozen=self.parent)
        self.assertEqual(selection["status"], "direction_stopped")
        self.summary("selection", selection)
        self.summary("calibrate", dict(status="direction_stopped", eligible=False, selection=selection))
        atomic_json(fixture.ledger, self.ledger)
        self.cache = fixture.root / "cpu-cache"
        manifest = seal(dict(schema_version="ditserve.research.cfg.cpu_embeddings.v1",
            model_manifest_sha256=fixture.manifest["sha256"], model_path=fixture.config["model_path"],
            revision=fixture.config["revision"]))
        atomic_json(self.cache / "manifest.json", manifest)
        self.binding = dict(root=str(self.cache), manifest_sha256=manifest["sha256"],
                            model_manifest_sha256=fixture.manifest["sha256"])
        mock = patch("ditserve.research.cfg_embeddings.verify_cache", return_value=manifest)
        mock.start()
        self.addCleanup(mock.stop)

    def unit(self, name, records, role, wall):
        with self.journal.unit(name, 1., time.monotonic() + 60):
            for i, record in enumerate(records):
                value = dict(status="completed", device_role=role, case_id=name + str(i), **{})
                value.update(record)
                self.journal.append(value)
        path = self.root / "units" / (name + ".json")
        value = read_json(path)
        value["ended_at"] = value["started_at"] + wall
        value.pop("sha256")
        atomic_json(path, seal(value))

    def summary(self, name, value):
        value = dict(value)
        value.pop("sha256", None)
        atomic_json(self.root / (name + ".json"), seal(dict(value,
            protocol_sha256=self.parent["sha256"], source_sha256=self.parent["source_sha256"])))

    def closed(self, phase, value, wall):
        self.summary(phase, dict(status="completed", **value))
        atomic_json(self.root / (phase + "-attempt.json"), seal(dict(status="completed", returncode=0,
            protocol_sha256=self.parent["sha256"])))
        role = phase.rsplit("-", 1)[1]
        atomic_json(self.root / (phase + "-job.json"), dict(phase=phase, session_id=phase,
            physical_devices=[dict(uuid="GPU-" + role)], protocol_sha256=self.parent["sha256"],
            source_sha256=self.parent["source_sha256"]))
        self.ledger["sessions"].append(dict(session_id=phase, status="completed", devices=["GPU-" + role],
            gpu_count=1, wall_seconds=wall, stage="correctness" if phase.startswith("correctness") else "calibration",
            protocol_sha256=self.parent["sha256"]))

    def build(self):
        return build_refinement(self.root, config=self.fx.config, ledger=self.fx.ledger)

    def freeze(self):
        self.fx.source.write_text("implementation = 'guarded_split_prefix'\n")
        return cfg_protocol.freeze_protocol(self.fx.config_path, self.fx.root / "results/run03", self.fx.ledger,
            model_manifest=self.fx.manifest, root=self.fx.root, refinement_run=self.root, cpu_embeddings=self.binding)

    def test_bound_failed_parent_does_not_grant_formal_qualification_or_repeat_independent(self):
        before = self.fx.ledger.read_bytes(), (self.root / "selection.json").read_bytes()
        child = self.freeze()
        self.assertEqual(child["prototype_continuation"], self.parent["prototype_continuation"])
        self.assertEqual(development_candidates(child), ["paired_008"])
        self.assertFalse(child["implementation_refinement"]["inherit_formal_evidence"])
        missing = select_candidate([], self.fx.config, "new-context-3", frozen=child)
        self.assertEqual(missing["status"], "unverified")
        self.assertEqual(missing["missing"], ["paired_008"])
        fresh = cache_rows("paired_008")
        for row in fresh:
            row["candidate"]["events"] = guard_events()
        self.assertEqual(select_candidate(fresh, self.fx.config, "ctx", frozen=child)["selected"], "paired_008")
        fresh[0]["candidate"]["events"][0]["cfg_research"]["split_prefix"]["cache_context_mode"] = "native_context"
        self.assertIsNone(select_candidate(fresh, self.fx.config, "ctx", frozen=child)["selected"])
        self.assertEqual(cfg_protocol.verify_protocol(self.fx.root / "results/run03/protocol.json", root=self.fx.root), child)
        self.assertEqual(before, (self.fx.ledger.read_bytes(), (self.root / "selection.json").read_bytes()))

    def test_units_sessions_and_stopped_selection_must_remain_complete(self):
        proof = self.build()
        ledger = copy.deepcopy(self.ledger)
        ledger["sessions"][0]["status"] = "failed"
        atomic_json(self.fx.ledger, ledger)
        with self.assertRaisesRegex(ValueError, "settled"):
            verify_refinement(proof, config=self.fx.config, ledger=self.fx.ledger)
        atomic_json(self.fx.ledger, self.ledger)
        unitpath = next((self.root / "units").glob("calibrate-*.json"))
        unit = read_json(unitpath)
        unit["performance_valid"] = False
        unit.pop("sha256")
        atomic_json(unitpath, seal(unit))
        with self.assertRaisesRegex(ValueError, "complete quality-passing"):
            self.build()

    def test_pending_predecessor_and_changed_session_fail_but_new_unrelated_session_is_allowed(self):
        proof = self.build()
        atomic_json(self.root / "pending-references.json", dict(pending=True))
        with self.assertRaisesRegex(ValueError, "pending work"):
            self.build()
        atomic_json(self.root / "pending-references.json", dict(pending=False))
        ledger = copy.deepcopy(self.ledger)
        ledger["active"] = dict(protocol_sha256="new-protocol")
        atomic_json(self.fx.ledger, ledger)
        self.assertEqual(verify_refinement(proof, config=self.fx.config, ledger=self.fx.ledger), proof)
        ledger["sessions"][0]["wall_seconds"] += 1
        atomic_json(self.fx.ledger, ledger)
        with self.assertRaisesRegex(ValueError, "closed predecessor"):
            verify_refinement(proof, config=self.fx.config, ledger=self.fx.ledger)

    def test_actual_cache_binding_required_and_historical_manifest_checked(self):
        with self.assertRaisesRegex(ValueError, "explicitly bound"):
            verify_cpu_binding(None, config=self.fx.config, model_manifest=self.fx.manifest)
        verify_cpu_binding(self.binding, config=self.fx.config, model_manifest=self.fx.manifest, verify_arrays=False)
        bad = dict(self.binding, manifest_sha256="changed")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            verify_cpu_binding(bad, config=self.fx.config, model_manifest=self.fx.manifest, verify_arrays=False)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            cfg_protocol.freeze_protocol(self.fx.config_path, self.fx.root / "new", self.fx.ledger,
                model_manifest=self.fx.manifest, root=self.fx.root, prototype_run=self.fx.parent_root,
                refinement_run=self.root, cpu_embeddings=self.binding)

    def test_cost_bound_uses_measured_full_units_and_estimated_installation_with_unchanged_caps(self):
        child = self.freeze()
        before = copy.deepcopy(self.ledger)
        plan = continuation_cost_plan(child, self.ledger)
        self.assertEqual(plan["status"], "fits")
        self.assertEqual(plan["active_candidates"], ["paired_008"])
        role = plan["per_device"]["a100"]
        self.assertEqual(role["formal_dev_pairs"], 16)
        self.assertEqual(role["held_out_pairs"], 4)
        self.assertEqual(role["measured_max_pair_wall_seconds"], 10.)
        self.assertEqual(role["estimated_installation_wall_seconds"], 5.)
        self.assertAlmostEqual(role["correctness_unit_estimate_wall_seconds"], (20 + 1.024) * 1.3)
        self.assertAlmostEqual(role["calibration_gpu_seconds"], (20 * 10 + 2 * (5 + 5)) * 1.3 + 60)
        pending = pending_cost_plan(child, self.fx.root / "results/run03", self.ledger, plan)
        self.assertAlmostEqual(pending["pending_stage_estimates_gpu_seconds"]["correctness"], plan["estimates_gpu_seconds"]["correctness"])
        self.assertAlmostEqual(pending["pending_stage_estimates_gpu_seconds"]["calibration"], plan["estimates_gpu_seconds"]["calibration"])
        self.assertEqual(self.ledger, before)
        self.ledger["reallocations"][0]["stage_allowances_gpu_seconds"]["correctness"] = 1.
        self.assertEqual(continuation_cost_plan(child, self.ledger)["status"], "budget_stopped")

    def test_fresh_three_record_unit_completes_correctness_without_using_shadow_as_formal_cost(self):
        child = self.freeze()
        root = self.fx.root / "results/run03"
        plan = continuation_cost_plan(child, self.ledger)
        journal = UnitJournal(root, child, ["GPU-a100"],
                              observe=lambda *a, **kw: dict(status="exclusive_observed"))
        records, summary = RefinementCorrectnessTests().evidence()
        # Slow instrumentation is a correctness cost, not a new hot pair observation.
        records[0]["continuous"]["elapsed_ms"] = 999000.
        unit = "correctness-a100-refinement-correctness-a100"
        with journal.unit(unit, 1., time.monotonic() + 60):
            for row in records:
                journal.append(row)
        atomic_json(root / "correctness-a100.json", seal(dict(summary, status="completed",
            protocol_sha256=child["sha256"], source_sha256=child["source_sha256"])))
        atomic_json(root / "correctness-a100-attempt.json", seal(dict(status="completed", returncode=0,
            protocol_sha256=child["sha256"])))
        rows = read_records(root, child)
        self.assertTrue(_completed(root, child, "correctness-a100", rows, ["paired_008"]))
        pending = pending_cost_plan(child, root, self.ledger, plan)
        self.assertIn("correctness-a100", pending["completed_phases"])
        self.assertEqual(pending["measured_per_device"]["a100"]["pair_wall_seconds"], 10.)
        self.assertAlmostEqual(pending["pending_stage_estimates_gpu_seconds"]["calibration"], plan["estimates_gpu_seconds"]["calibration"])
        self.assertFalse(_completed(root, child, "correctness-a100", rows[:2], ["paired_008"]))


class RefinementCorrectnessTests(unittest.TestCase):
    def evidence(self):
        def run(impl, action, diagnostic):
            return dict(identity=dict(implementation=impl), action=action, diagnostic=diagnostic,
                events=guard_events(diagnostic),
                diagnostic_scope="same_input_prediction_only_correctness" if diagnostic else None,
                latent_sha256="same-latent", output_sha256="same-output",
                traces=[dict(step=i, state_sha256=str(i)) for i in range(1, 33)])
        diagnostic, restored, oracle = run("split_prefix", "stay", True), run("split_prefix", "local_full_migrate", False), run("oracle", "stay", False)
        common = dict(status="completed", performance_valid=True, unit_id="unit", recipe_id="paired_008", device_role="a100", equal=True, timing_eligible=False)
        rows = [dict(common, type="cfg_correctness", continuous=diagnostic, restored=restored),
                dict(common, type="cfg_implementation_equivalence", continuous=oracle, restored=diagnostic),
                dict(common, type="cfg_diagnostic_neutrality", clean=restored, instrumented=diagnostic,
                     off_trajectory_latent_error_measured=False)]
        summary = dict(checks_passed=True, diagnostic_gpu_validated=True, validation_mode="refinement_three_trajectories")
        return rows, summary

    def test_three_trajectories_required_and_summary_cannot_override_missing_or_different_state(self):
        rows, summary = self.evidence()
        self.assertTrue(correctness_complete(rows, summary, "a100"))
        self.assertFalse(correctness_complete(rows[:2], summary, "a100"))
        rows[0]["restored"]["traces"][16]["state_sha256"] = "changed"
        self.assertFalse(correctness_complete(rows, summary, "a100"))
        rows, summary = self.evidence()
        rows[1]["continuous"]["traces"] = rows[1]["continuous"]["traces"][:-1]
        self.assertFalse(correctness_complete(rows, summary, "a100"))

    def test_guard_mode_cost_and_actual_shadow_evidence_are_required(self):
        for mutation in ("mode", "negative", "nan", "missing", "shadow"):
            with self.subTest(mutation=mutation):
                rows, summary = self.evidence()
                run = rows[0]["continuous"]
                detail = run["events"][0]["cfg_research"]["split_prefix"]
                if mutation == "mode":
                    detail["cache_context_mode"] = "native_context"
                elif mutation == "negative":
                    detail["context_guard_ms"] = -1
                elif mutation == "nan":
                    detail["context_guard_ms"] = float("nan")
                elif mutation == "missing":
                    run["events"].pop()
                else:
                    for event in run["events"]:
                        event["cfg_research"]["same_input_prediction_error"] = None
                self.assertFalse(correctness_complete(rows, summary, "a100"))


def guard_events(shadow=False):
    return [dict(kind="step", cfg_research=dict(split_prefix=dict(cache_context_mode="noop_guarded_v1",
        context_guard_ms=.01), same_input_prediction_error={"cfg": {"relative_l2": .01}} if shadow else None))
        for _ in range(32)]


if __name__ == "__main__":
    unittest.main()
