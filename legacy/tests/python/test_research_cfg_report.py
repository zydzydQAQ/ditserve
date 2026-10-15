import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, file_digest
from ditserve.research.dual_matrix import seal
from ditserve.research.cfg_report import (asymmetry_evidence, cost_memory_summary, descriptive_inputs,
                                        evaluate_hypotheses, make_figures, report)
from ditserve.research.retention import apply_plan, build_plan, verify_plan
from test_research_cfg_matrix import (CANDIDATES, MIGRATION_ACTIONS, cache_rows, config,
                                      mechanism_pairs, migration_rows, quality)


def qualified_cache():
    return sum((cache_rows(p, "stage_a") + cache_rows(p) for p in CANDIDATES), []) + cache_rows(phase="holdout")


class CFGReportTests(unittest.TestCase):
    def test_empty_evidence_is_missing_not_negative_and_four_plots_show_missing(self):
        value = evaluate_hypotheses([], config())
        self.assertEqual(value["cache"]["status"], "insufficient_evidence")
        self.assertEqual(value["migration"]["status"], "insufficient_evidence")
        self.assertEqual(value["sp"]["status"], "deferred")
        with tempfile.TemporaryDirectory() as temp:
            figures = make_figures(temp, [])
            if figures["status"] == "unavailable":
                self.skipTest(figures["reason"])
            self.assertEqual(len(figures["paths"]), 4)
            for figure in figures["paths"]:
                self.assertFalse(figure["measured"])
                self.assertIn("Insufficient completed-unit evidence", (Path(temp) / figure["path"]).read_text())

    def test_supported_cache_still_cannot_establish_controller_or_real_memory_benefit(self):
        value = evaluate_hypotheses(qualified_cache(), config())
        self.assertEqual(value["cache"]["status"], "supported")
        self.assertEqual(value["cache"]["selected"], "independent_004")
        self.assertEqual(value["cache_joint_control"]["status"], "insufficient_evidence")
        self.assertEqual(value["memory"]["status"], "insufficient_evidence")

    def test_holdout_failure_stops_chosen_policy_without_switching_to_other_candidate(self):
        rows = qualified_cache() + cache_rows("paired_008", "holdout")
        row = next(r for r in rows if r["phase"] == "holdout" and r["recipe_id"] == "independent_004")
        row["quality"] = quality(row["candidate"], row["reference"], .06)
        value = evaluate_hypotheses(rows, config())
        self.assertEqual(value["selection"]["selected"], "independent_004")
        self.assertEqual(value["cache"]["status"], "falsified")

    def test_migration_reports_each_action_and_rechecks_real_queue_coverage(self):
        rows = qualified_cache()
        rows += sum((migration_rows(action=a, split=s) for a in MIGRATION_ACTIONS for s in ("dev", "holdout")), [])
        rows += mechanism_pairs("busy")
        value = evaluate_hypotheses(rows, config())
        self.assertEqual(value["migration"]["status"], "supported")
        rows[-1]["queue_wait_ms"] = 124.
        value = evaluate_hypotheses(rows, config())
        self.assertEqual(value["migration"]["status"], "insufficient_evidence")
        self.assertFalse(value["mechanism"]["busy"]["queue_coverage_verified"])

    def test_one_failed_migration_does_not_block_independently_qualified_other_action(self):
        rows = qualified_cache()
        rows += sum((migration_rows(action=a, split=s) for a in MIGRATION_ACTIONS for s in ("dev", "holdout")), [])
        failed = next(r for r in rows if r.get("type") == "cfg_migration_quality" and r["action"] == "full_migrate")
        failed["path_quality"] = quality(failed["candidate"], failed["path_reference"], .06)
        rows += mechanism_pairs(actions=("cold_migrate",))
        value = evaluate_hypotheses(rows, config())
        self.assertEqual(value["migration"]["status"], "supported")
        self.assertFalse(value["migration_actions"]["full_migrate"]["eligible"])
        self.assertTrue(value["migration_actions"]["cold_migrate"]["eligible"])

    def test_oracle_and_diagnostic_steps_never_enter_formal_cost_memory_plot(self):
        rows = cache_rows("paired_008", "stage_a")
        rows += [dict(type="cfg_diagnostic", case_id="d", performance_valid=True, timing_eligible=False,
                     run=dict(events=[dict(kind="step", step_index=1, cfg_research=dict(
                         same_input_prediction_error={k: dict(relative_l2=.1) for k in ("cond", "uncond", "cfg")}))]))]
        with tempfile.TemporaryDirectory() as temp:
            result = make_figures(temp, rows)
            if result["status"] == "unavailable":
                self.skipTest(result["reason"])
            by_name = {Path(x["path"]).stem: x for x in result["paths"]}
            self.assertTrue(by_name["cfg-branch-error"]["measured"])
            self.assertTrue(by_name["quality-time"]["measured"])
            self.assertTrue(by_name["cost-memory"]["measured"])
            self.assertEqual(by_name["cost-memory"]["formal_samples"], 0)
            self.assertGreater(by_name["cost-memory"]["prototype_samples"], 0)
            svg = (Path(temp) / by_name["cost-memory"]["path"]).read_text()
            self.assertIn("Formal performance data missing", svg)
            self.assertIn("descriptive costs only", svg)

    def test_prototype_overhead_and_peak_are_descriptive_and_replay_is_not_subtracted_twice(self):
        rows = cache_rows("paired_008", "stage_a")[:1]
        candidate = rows[0]["candidate"]
        candidate["peak_memory_bytes"] = dict(allocated=2**30, reserved=2**31)
        candidate["events"] = [dict(kind="step", cfg_research=dict(replay_ms=50.))]
        rows[0]["reference"].pop("peak_memory_bytes", None)
        summary = cost_memory_summary(rows)
        oracle = next(s for s in summary["samples"] if s["implementation"] == "oracle")
        self.assertEqual(oracle["optimistic_ms"], oracle["elapsed_ms"] - 5. - 10.)
        self.assertEqual(oracle["accepted_replay_ms"], 50.)
        self.assertEqual(oracle["allocated_peak_bytes"], 2**30)
        self.assertFalse(oracle["formal_cost_eligible"])
        self.assertFalse(oracle["performance_qualified"])
        off = next(g for g in summary["groups"] if g["recipe_id"] == "off")
        self.assertIsNone(off["snapshot_ms"]["mean"])
        self.assertIsNone(off["allocated_peak_bytes"]["mean"])
        candidate["optimistic_bound"]["optimistic_ms"] = 1.
        self.assertIsNone(next(s for s in cost_memory_summary(rows)["samples"] if s["implementation"] == "oracle")["optimistic_ms"])

    def test_formal_speculative_gate_overhead_is_included_without_discounting_elapsed(self):
        rows = cache_rows("paired_008")[:1]
        rows[0]["candidate"]["events"] = [dict(kind="step", cfg_research=dict(
            split_prefix=dict(gate_probe_ms=3., prefix_ms=dict(cond=9., uncond=8.))))]
        summary = cost_memory_summary(rows)
        candidate = next(s for s in summary["samples"] if s["recipe_id"] == "paired_008")
        self.assertEqual(candidate["elapsed_ms"], 90.)
        self.assertEqual(candidate["gate_probe_ms"], 3.)
        self.assertIsNone(candidate["optimistic_ms"])
        self.assertTrue(candidate["formal_cost_eligible"])

    def diagnostic_comparison(self):
        independent = cache_rows("independent_008", "stage_a")[0]
        independent.update(type="cfg_diagnostic", phase="diagnostic", timing_eligible=False)
        independent["candidate"]["diagnostic"] = True
        independent["quality"] = quality(independent["candidate"], independent["reference"], .07)
        events = []
        for step, path, loss in ((1, "CC", None), (2, "CR", .2), (3, "RR", .1)):
            gates = [dict(branch=b, native_reuse=p == "R", accepted_reuse=p == "R") for b, p in zip(("cond", "uncond"), path)]
            events.append(dict(kind="step", completed_steps=step, cfg_research=dict(
                native_attempt=dict(gates=gates), accepted_pair=dict(gates=gates),
                same_input_prediction_error=None if loss is None else dict(cfg=dict(relative_l2=loss)))))
        independent["candidate"]["events"] = events
        paired = cache_rows("paired_008", "stage_a")[0]
        paired["quality"] = quality(paired["candidate"], paired["reference"], .02)
        return [independent, paired]

    def test_asymmetry_uses_same_latent_samples_and_same_reference_quality_pairs_without_causal_claim(self):
        rows = self.diagnostic_comparison()
        summary = asymmetry_evidence(rows, config())
        self.assertEqual(summary["paired_quality_comparisons"], 1)
        self.assertEqual(summary["mixed_error_higher_trajectories"], 1)
        self.assertEqual(summary["paired_quality_improved_inputs"], 1)
        sample = summary["samples"][0]
        self.assertEqual(sample["mixed_steps"], 1)
        self.assertEqual(len(sample["guided_error_samples"]), 2)
        self.assertAlmostEqual(sample["paired_comparison"]["lpips_reduction"], .05)
        self.assertFalse(summary["causal_identification"])
        self.assertFalse(summary["matched_cost_control"])
        self.assertFalse(summary["performance_qualified"])

    def test_asymmetry_never_pairs_unbound_changed_reference_or_different_context(self):
        rows = self.diagnostic_comparison()
        rows[1]["reference"]["output_sha256"] = "f" * 64
        rows[1]["quality"] = quality(rows[1]["candidate"], rows[1]["reference"], .02)
        self.assertEqual(asymmetry_evidence(rows, config())["paired_quality_comparisons"], 0)
        rows = self.diagnostic_comparison()
        rows[0]["quality"]["candidate_output_sha256"] = "e" * 64
        self.assertEqual(asymmetry_evidence(rows, config())["independent_trajectories"], 0)
        rows = self.diagnostic_comparison()
        rows[1]["context_sha256"] = "other"
        self.assertEqual(asymmetry_evidence(rows, config())["paired_quality_comparisons"], 0)

    def test_verified_parent_is_descriptive_only_and_never_replaces_current_formal_data(self):
        from test_research_cfg_continuation import CfgContinuationTests
        fixture = CfgContinuationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        child = fixture.new_freeze()
        current = fixture.current_development()
        parent_bytes = (fixture.parent_root / "records.jsonl").read_bytes()
        result = evaluate_hypotheses(current, fixture.config, frozen=child)
        self.assertEqual(result["selection"]["selected"], "independent_004")
        self.assertEqual(result["cache"]["status"], "insufficient_evidence")  # No current holdout.
        descriptions = descriptive_inputs(child, current)
        parents = [r for r in descriptions if r["descriptive_source"]["origin"] == "parent_prototype"]
        self.assertEqual(len(parents), 12)
        self.assertEqual({r["type"] for r in parents}, {"cfg_screen"})
        self.assertTrue(all(r["timing_eligible"] is False for r in parents))
        self.assertEqual({r["descriptive_source"]["source_sha256"] for r in parents}, {fixture.parent["source_sha256"]})
        self.assertEqual((fixture.parent_root / "records.jsonl").read_bytes(), parent_bytes)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            evaluate_hypotheses(parents, fixture.config, frozen=child)
        forged = copy.deepcopy(child)
        forged["prototype_continuation"]["parent_source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            descriptive_inputs(forged, current)

    def test_multiple_contexts_cannot_reuse_qualification(self):
        rows = qualified_cache()
        rows[0]["context_sha256"] = "other"
        self.assertEqual(evaluate_hypotheses(rows, config())["cache"]["status"], "insufficient_evidence")

    def test_mechanism_cannot_reuse_another_policy_cost_qualification(self):
        rows = qualified_cache()
        rows += sum((migration_rows(action=a, split=s) for a in MIGRATION_ACTIONS for s in ("dev", "holdout")), [])
        pairs = mechanism_pairs()
        for row in pairs:
            row["recipe_id"] = "paired_008"
        value = evaluate_hypotheses(rows + pairs, config())
        self.assertEqual(value["migration"]["status"], "insufficient_evidence")

    def test_report_accepts_declared_pruning_and_rejects_undeclared_missing_without_rewriting_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            ledger = root / "gpu-budget.json"
            atomic_json(ledger, dict(active=None, used_gpu_seconds=321., limit_gpu_seconds=28800.))
            frozen = dict(sha256="p", source_sha256="s", ledger=str(ledger), config=config())
            atomic_json(root / "protocol.json", frozen)
            raw = root / "sample.npy"
            raw.write_bytes(b"test original array")
            raw_hash = file_digest(raw)
            row = seal(dict(type="probe", unit_id="u", case_id="ordinary", protocol_sha256="p", source_sha256="s",
                            run=dict(output_path="sample.npy", output_sha256=raw_hash)))
            original_records = json.dumps(row) + "\n"
            (root / "records.jsonl").write_text(original_records)
            observations = [seal(dict(phase=phase, status="exclusive_observed", unit_id="u", protocol_sha256="p", source_sha256="s"))
                            for phase in ("start", "completed")]
            atomic_json(root / "units/u.json", seal(dict(status="completed", unit_id="u", performance_valid=True,
                protocol_sha256="p", source_sha256="s", records=[row["sha256"]], observations=observations)))
            case = dict(series="test", run=str(root), ended_at=1, case_id="ordinary", kind="probe", split="unknown",
                policy=None, selected_policy=None, quality=None, failed=False, verified=True, evaluated=True,
                artifacts=[dict(path=str(raw), sha256=raw_hash)], reference_sha256s=[])
            plan = build_plan(root, Path(temp) / "key", catalog=dict(cases=[case], sources=[]))
            apply_plan(plan, verify_plan(plan))
            with patch("ditserve.research.cfg_report.verify_protocol", return_value=frozen), \
                 patch("ditserve.research.cfg_report.make_figures", return_value=dict(status="unavailable", paths=[])):
                value = report(root)
                self.assertEqual(value["evidence_availability"]["retained_metrics_only"], 1)
                self.assertEqual(value["budget"]["settled_gpu_seconds"], 321.)
                self.assertEqual((root / "records.jsonl").read_text(), original_records)
                (root / "retention-index.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    report(root)

    def test_report_refuses_active_ledger_before_archive_or_raw_hashing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "gpu-budget.json"
            atomic_json(ledger, dict(active=dict(pid=1)))
            atomic_json(root / "protocol.json", dict(ledger=str(ledger)))
            with patch("ditserve.research.cfg_report.verify_protocol", side_effect=AssertionError("hashing forbidden")):
                with self.assertRaisesRegex(RuntimeError, "unsettled"):
                    report(root)


if __name__ == "__main__":
    unittest.main()
