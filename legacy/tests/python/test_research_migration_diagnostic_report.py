import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research.migration_diagnostic_experiment import ACTIONS, recompute_checks
from ditserve.research.migration_diagnostic_protocol import DEFAULT_CONFIG
from ditserve.research.migration_diagnostic_report import analyze_rows, make_figures, report


def diagnostic_row(comparison="original", seed=102, order="forward"):
    diagnostic = read_json(DEFAULT_CONFIG)
    backends = diagnostic["attention_by_comparison"][comparison]
    context = "context-" + comparison

    def run(action):
        role = "v100" if action == "full_migrate" or action.startswith("v100") else "a100"
        route = ["v100", "a100"] if action == "full_migrate" else [role]
        def tag(step):
            return ("migrated" if action == "full_migrate" and step > 16 else role) + "/" + str(seed)
        signatures = {str(step): dict(step_index=step, state_sha256=digest((tag(step), step)))
                      for step in (1, 16, 17, 32)}
        return dict(identity=dict(context_sha256=context, recipe="off", action=action, implementation="native", boundary=16,
            path=route, attention_path=[backends[r] for r in route], input=dict(prompt_id="kettle", seed=seed)),
            signatures=signatures, traces=[dict(step=s, state_sha256=digest((tag(s), s))) for s in range(1, 33)],
            boundary_signature=signatures["16"], boundary_equal=True,
            latent_sha256=digest((tag(32), "latent")), output_sha256=digest((tag(32), "output")), elapsed_ms=10000.)

    runs = {action: run(action) for action in ACTIONS}
    before = runs["v100_stay"]["signatures"]["16"]
    source = dict(before=before, after=runs["v100_stay"]["signatures"]["17"],
                  target_device="GPU-v100", backend=backends["v100"], snapshot_sha256="snapshot")
    target = dict(before=before, after=runs["full_migrate"]["signatures"]["17"],
                  target_device="GPU-a100", backend=backends["a100"], snapshot_sha256="snapshot")
    shadow = dict(source=source, target=target, source_status=dict(residency="diagnostic_paused", signature=before),
        snapshot_sha256="snapshot", step=17,
        errors={key:dict(relative_l2=0. if key == "latent_before" else .01 if comparison == "original" else .004,
            max_abs=0. if key == "latent_before" else .02) for key in ("cond", "uncond", "cfg", "latent_before", "latent_after")})
    qualities = {}
    for name, candidate, reference in (("v100_vs_a100", "v100_stay", "a100_stay"),
            ("migration_vs_a100", "full_migrate", "a100_stay"), ("migration_vs_v100", "full_migrate", "v100_stay")):
        loss = .076 if seed == 102 else .0005
        if comparison == "math":
            loss /= 4
        qualities[name] = dict(status="ok", lpips_status="ok", finite=True, range_valid=True, frame_count=17,
            lpips_mean=loss, lpips_p95=loss, lpips_max=loss, lpips_per_frame=[loss] * 17, temporal_delta_rmse=.01,
            candidate_output_sha256=runs[candidate]["output_sha256"], reference_output_sha256=runs[reference]["output_sha256"],
            candidate_identity_sha256=digest(runs[candidate]["identity"]), reference_identity_sha256=digest(runs[reference]["identity"]))
    row = dict(type="migration_diagnostic", comparison=comparison, prompt_id="kettle", seed=seed, order=order,
        case_id="%s-%s-%s" % (comparison, seed, order), context_sha256=context, recipe_id="off", status="completed",
        timing_eligible=False, performance_qualified=False, quality_qualified=False, performance_valid=True,
        runs=runs, shadow=shadow, qualities=qualities, checks=recompute_checks(runs, shadow), checks_passed=True)
    return dict(row, sha256=digest(row))


def all_rows():
    return [diagnostic_row(comparison, seed, order) for comparison in ("original", "math")
            for seed in (101, 102) for order in ("forward", "reverse")]


class MigrationDiagnosticReportTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(DEFAULT_CONFIG)

    def test_complete_controls_describe_reproduction_without_granting_qualification(self):
        result = analyze_rows(all_rows(), self.config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["failure_reproduction"]["status"], "reproduced_in_both_orders")
        self.assertEqual(result["completed_units"], 8)
        self.assertEqual(len(result["backend_comparison"]), 4)
        self.assertTrue(all(v["all_hashes_equal"] for v in result["repeatability"]))
        self.assertGreater(result["backend_comparison"][0]["lpips"]["migration_vs_a100"]["original_minus_math"], 0.)
        for key in ("migration_qualified", "performance_qualified", "quality_qualified"):
            self.assertIs(result[key], False)
        self.assertIn("不能单独区分", result["attribution"])

    def test_original_reproduction_remains_visible_when_math_is_unmeasured(self):
        result = analyze_rows(all_rows()[:4], self.config)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["phases"]["math"]["status"], "unverified")
        self.assertEqual(result["failure_reproduction"]["status"], "reproduced_in_both_orders")
        self.assertEqual(result["backend_comparison"], [])

    def test_local_restore_failure_is_invalid_even_when_recorded_flags_claim_pass(self):
        rows = all_rows()
        rows[0]["runs"]["a100_local_restore"]["latent_sha256"] = "wrong"
        result = analyze_rows(rows, self.config)
        self.assertEqual(result["status"], "diagnostic_invalid")
        self.assertTrue(result["invalid_controls"])

    def test_source_mutation_and_wrong_shadow_device_are_rejected(self):
        for mutate in (lambda row: row["shadow"]["source_status"].update(residency="gpu"),
                       lambda row: row["shadow"]["target"].update(target_device="GPU-v100")):
            rows = all_rows()
            mutate(rows[0])
            with self.subTest(mutate=mutate):
                self.assertEqual(analyze_rows(rows, self.config)["status"], "diagnostic_invalid")

    def test_quality_must_reference_the_exact_outputs_and_inputs(self):
        rows = all_rows()
        rows[0]["qualities"]["migration_vs_a100"]["candidate_identity_sha256"] = "other-trajectory"
        self.assertEqual(analyze_rows(rows, self.config)["status"], "diagnostic_invalid")

    def test_backend_label_cannot_hide_a_different_native_trajectory(self):
        rows = all_rows()
        rows[0]["runs"]["v100_stay"]["identity"]["attention_path"] = ["math"]
        self.assertEqual(analyze_rows(rows, self.config)["status"], "diagnostic_invalid")

    def test_duplicate_and_invalidated_units_do_not_complete_the_matrix(self):
        rows = all_rows()
        self.assertEqual(analyze_rows(rows + [copy.deepcopy(rows[0])], self.config)["status"], "diagnostic_invalid")
        rows[0]["performance_valid"] = False
        result = analyze_rows(rows, self.config)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["completed_units"], 7)
        self.assertEqual(len(result["excluded_units"]), 1)

    def test_repeatability_is_recomputed_instead_of_trusting_summary(self):
        rows = all_rows()
        # Change both local-equivalent outputs and refresh their quality bindings.
        row = rows[3]
        for action in ("a100_stay", "a100_local_restore"):
            row["runs"][action]["output_sha256"] = "repeat-drift"
        for quality in row["qualities"].values():
            if quality["reference_output_sha256"] == rows[2]["runs"]["a100_stay"]["output_sha256"]:
                quality["reference_output_sha256"] = "repeat-drift"
        row["repeatability"] = {action: dict(output_equal=True, latent_equal=True) for action in ACTIONS}
        result = analyze_rows(rows, self.config)
        self.assertEqual(result["status"], "completed")
        sample = next(v for v in result["repeatability"] if v["comparison"] == "original" and v["seed"] == 102)
        self.assertFalse(sample["all_hashes_equal"])
        self.assertFalse(sample["actions"]["a100_stay"]["output_equal"])

    def test_empty_report_keeps_missing_observations_distinct_from_zero(self):
        result = analyze_rows([], self.config)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["samples"], [])
        self.assertEqual(result["failure_reproduction"]["original_lpips_by_order"], dict(forward=None, reverse=None))

    def test_report_writes_budget_and_declared_retention_availability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "budget.json"
            atomic_json(ledger, dict(active=None, limit_gpu_seconds=28800., used_gpu_seconds=15000.,
                stage_limits_gpu_seconds=dict(migration=5600.), stage_used_gpu_seconds=dict(migration=600.),
                sessions=[dict(protocol_sha256="new-protocol", gpu_seconds=200.)]))
            frozen = dict(ledger=str(ledger), sha256="new-protocol", source_sha256="new-source", diagnostic=self.config,
                          source_archive={}, parent_failure={})
            atomic_json(root / "protocol.json", frozen)
            rows = all_rows()
            rows[0]["runs"]["a100_stay"]["output_path"] = "arrays/example.npy"
            with patch("ditserve.research.migration_diagnostic_report.verify_protocol", return_value=frozen), \
                 patch("ditserve.research.migration_diagnostic_report.read_records", return_value=rows), \
                 patch("ditserve.research.migration_diagnostic_report.verify_reference", return_value=dict(status="retained_metrics_only")), \
                 patch("ditserve.research.migration_diagnostic_report.make_figures", return_value=dict(status="completed", paths=[])):
                result = report(root)
            self.assertEqual(result["budget"]["run_gpu_seconds"], 200.)
            self.assertEqual(result["budget"]["remaining_migration_gpu_seconds"], 5000.)
            self.assertEqual(result["evidence_availability"]["retained_metrics_only"], 1)
            self.assertTrue((root / "report.md").is_file())
            self.assertEqual(read_json(root / "report.json"), result)

    def test_four_scientific_figures_render_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = make_figures(root, analyze_rows(all_rows(), self.config))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["paths"]), 4)
            self.assertTrue(all((root / item["path"]).is_file() for item in result["paths"]))


if __name__ == "__main__":
    unittest.main()
