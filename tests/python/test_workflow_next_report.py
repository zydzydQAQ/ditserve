"""Result eligibility must not promote diagnostic or incomplete evidence."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_next_report as report


def row(identifier="r0"):
    return dict(request_id=identifier, status="completed", kind="video",
                spec=dict(prompt="boat", seed=1), arrival_ms=0, deadline_ms=100, finish_ms=50,
                outputs=[dict(output=dict(finite=True, array_sha256="video"),
                              latent=dict(finite=True, array_sha256="latent"))])


class PairTests(unittest.TestCase):
    def test_encoding_comparisons_keep_all_predeclared_pairs(self):
        def result(names):
            return dict(phases=[dict(id=name, cells=[{}]) for name in names])
        self.assertEqual(report.native_comparison_pairs(result(["fused", "disabled", "staged"])),
                         [("disabled", "staged"), ("disabled", "fused"), ("staged", "fused")])
        self.assertEqual(report.native_comparison_pairs(result(["enabled", "disabled"])),
                         [("disabled", "enabled")])

    def test_native_digest_aliases_and_video_latents_are_both_required(self):
        left, right = [row()], [row()]
        self.assertTrue(report.exact_pair(left, right)["passed"])
        del right[0]["outputs"][0]["latent"]
        self.assertFalse(report.exact_pair(left, right)["passed"])
        prefetch = dict(request_id="r0", spec={}, outputs=dict(
            latent=dict(finite=True, sha256="latent")))
        self.assertFalse(report.exact_pair([prefetch], [prefetch])["passed"])

    def test_identically_truncated_pairs_fail_against_frozen_workload(self):
        observed = [row()]
        frozen = [row(), row("r1")]
        result = dict(config=dict(phases=[dict(id=name, workload=frozen) for name in ("reference", "candidate")]),
                      phases=[dict(id=name, cells=[dict(status="completed", requests=observed, elapsed_ms=50)])
                              for name in ("reference", "candidate")])
        comparison = report.comparison(result, "reference", "candidate")
        self.assertTrue(comparison["numeric"]["passed"])
        self.assertFalse(comparison["complete_frozen_requests"])
        self.assertFalse(comparison["eligible"])

    def test_native_injected_spec_request_id_matches_frozen_workload(self):
        frozen = row()
        observed = copy.deepcopy(frozen)
        observed["spec"]["request_id"] = observed["request_id"]
        result = dict(config=dict(phases=[dict(id=name, workload=[copy.deepcopy(frozen)])
                                         for name in ("reference", "candidate")]),
                      phases=[dict(id=name, cells=[dict(status="completed",
                              requests=[copy.deepcopy(observed)], elapsed_ms=50)])
                              for name in ("reference", "candidate")])
        comparison = report.comparison(result, "reference", "candidate")
        self.assertTrue(comparison["complete_frozen_requests"])
        self.assertTrue(comparison["eligible"])
        self.assertNotIn("request_id", result["config"]["phases"][0]["workload"][0]["spec"])
        for changes in (dict(request_id="different"), dict(prompt="different"), dict(seed=2)):
            with self.subTest(changes=changes):
                modified = copy.deepcopy(result)
                # Both observed arms changing together must still fail against
                # the frozen input, even when their output hashes are identical.
                for phase in modified["phases"]:
                    phase["cells"][0]["requests"][0]["spec"].update(changes)
                comparison = report.comparison(modified, "reference", "candidate")
                self.assertFalse(comparison["complete_frozen_requests"])
                self.assertFalse(comparison["eligible"])

    def test_inconsistent_nested_request_id_fails_without_planned_workload(self):
        observed = row()
        observed["spec"]["request_id"] = "different"
        result = dict(phases=[dict(id=name, cells=[dict(status="completed",
                      requests=[copy.deepcopy(observed)], elapsed_ms=50)])
                      for name in ("reference", "candidate")])
        comparison = report.comparison(result, "reference", "candidate")
        self.assertFalse(comparison["same_trajectory"])
        self.assertFalse(comparison["eligible"])

    def test_diagnostic_timings_never_gain_performance_qualification(self):
        for result in (dict(performance_evidence=False),
                       dict(schema="ditserve.workflow.elastic_validation.v1"),
                       dict(schema="ditserve.workflow.serving.sp.v1"),
                       dict(memory_diagnostics=True)):
            self.assertFalse(report.performance_qualified(
                dict(result=result, qualified=True, capacity=True), {}))
        self.assertFalse(report.performance_qualified(
            dict(result={}, qualified=True, capacity=True), dict(profiling=True)))

    def test_explicit_failed_numerical_gate_is_not_merely_unverified(self):
        self.assertEqual(report.numeric_status(dict(guarded=True, qualified=False,
            result=dict(qualification=dict(passed=False)))), "failed")
        self.assertEqual(report.numeric_status(dict(guarded=False, qualified=False,
            result=dict(qualification=dict(passed=False)))), "unverified")
        self.assertEqual(report.numeric_status(dict(guarded=True, qualified=False,
            result=dict(status="capacity_failed", qualification=dict(passed=False)))), "unverified")

    def test_elastic_hash_differences_do_not_claim_online_tolerance_or_numeric_failure(self):
        reference, candidate = row(), row()
        candidate["outputs"][0]["latent"]["array_sha256"] = "different"
        result = dict(phases=[dict(id=name, cells=[dict(
            status="completed", requests=[value], elapsed_ms=50)])
            for name, value in (("step", reference), ("block", candidate))])
        result = report.comparison(result, "step", "block", elastic=True,
                                   numeric_gate_linkage=dict(passed=True))
        self.assertFalse(result["eligible"])
        self.assertEqual(result["numeric"]["status"], "unverified")
        self.assertEqual(result["numeric"]["tolerance_status"], "unverified")
        self.assertEqual(result["throughput_evidence"], "observational")
        self.assertTrue(result["technical_feasibility"])

    def test_elastic_exact_hashes_still_require_all_dedicated_gates(self):
        result = dict(phases=[dict(id=name, cells=[dict(
            status="completed", requests=[row()], elapsed_ms=50)])
            for name in ("step", "step-reserved", "block")])
        self.assertEqual(report.native_comparison_pairs(result),
                         [("step", "block"), ("step", "step-reserved")])
        self.assertFalse(report.comparison(result, "step", "block", elastic=True)["eligible"])
        self.assertTrue(report.comparison(result, "step", "block", elastic=True,
                                         numeric_gate_linkage=dict(passed=True))["eligible"])
        linkage = report.elastic_gate_linkage(dict(jobs=[]))
        self.assertFalse(linkage["passed"])
        self.assertEqual(len(linkage["gates"]), 5)


class EvidenceTests(unittest.TestCase):
    def evidence(self, result, *, samples=2, errors=None, metadata=None, frozen_config=None,
                 configuration_sha256=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "measurements" / "job"
            out.mkdir(parents=True)
            path = out / "run.json"
            path.write_text(json.dumps(result))
            launch = root / "launches" / "job" / "launch.json"
            launch.parent.mkdir(parents=True)
            launch.write_text(json.dumps(dict(
                status="completed", output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                gpu_monitor=dict(errors=errors or [], conflicts=[], memory={"0": dict(
                    samples=samples, minimum_free_bytes=2*1024**3, total_bytes=24*1024**3)}))))
            job = dict(id="job", output=str(out), devices=[0], metadata=metadata or {})
            if frozen_config is not None:
                input_path = root / "input.json"
                input_path.write_text(json.dumps(frozen_config))
                job.update(input=str(input_path), configuration_sha256=(configuration_sha256 or
                           report.digest(frozen_config)))
            return report.evidence(dict(root=str(root)), job)

    def test_monitor_errors_and_single_sample_never_pass_capacity_or_guard(self):
        self.assertFalse(self.evidence(dict(status="completed"), samples=1)["capacity"])
        self.assertFalse(self.evidence(dict(status="completed"), errors=["lost monitor"])["guarded"])

    def test_encoding_numeric_requires_independent_audit_and_frozen_input(self):
        result = dict(status="completed", qualification=dict(passed=True))
        target = "ditserve.evaluation.workflow_encoding_optimization_report.audit_result"
        capacity = "ditserve.evaluation.workflow_encoding_optimization_report.raw_capacity"
        with patch(target, return_value=dict(passed=False)), patch(capacity, return_value=dict(passed=True)):
            self.assertFalse(self.evidence(result, metadata=dict(role="encoding_numeric"),
                                           frozen_config={})["qualified"])
        with patch(target, return_value=dict(passed=True)), patch(capacity, return_value=dict(passed=True)):
            self.assertTrue(self.evidence(result, metadata=dict(role="encoding_numeric"),
                                          frozen_config={})["qualified"])
            self.assertFalse(self.evidence(result, metadata=dict(role="encoding_numeric"),
                                            frozen_config={}, configuration_sha256="changed")["qualified"])
        with patch(target, return_value=dict(passed=True)), patch(capacity, return_value=dict(passed=False)):
            self.assertFalse(self.evidence(result, metadata=dict(role="encoding_numeric"),
                                           frozen_config={})["qualified"])

    def test_encoding_online_cannot_gain_qualification_without_numeric_dependencies(self):
        with patch("ditserve.evaluation.workflow_encoding_optimization_report.audit_result",
                   return_value=dict(passed=True)):
            ev = self.evidence(dict(status="completed"), metadata=dict(role="encoding_online"), frozen_config={})
            self.assertFalse(ev["qualified"])
            self.assertEqual(ev["encoding_audit"]["dependencies"], {})

    def test_unconfigured_native_gate_is_not_a_failed_measurement_or_numeric_pass(self):
        qualification = dict(qualified=False, status="unverified", checks=[],
                             scopes=dict(numeric=dict(status="unverified")))
        result = dict(status="completed", phases=[dict(status="completed", qualification=qualification)])
        ev = self.evidence(result)
        self.assertTrue(ev["qualified"])
        self.assertEqual(report.numeric_status(ev), "unverified")
        qualification.update(qualified=True, status="passed",
                             checks=[dict(scope="performance", status="passed")])
        ev = self.evidence(result)
        self.assertTrue(ev["qualified"])
        self.assertEqual(report.numeric_status(ev), "unverified")
        qualification.update(qualified=False, status="failed")
        self.assertFalse(self.evidence(result)["qualified"])

    def test_elastic_requires_all_configured_cases_and_real_stage_gates(self):
        case = dict(seed=1, frames=17, passed=True, cross_card_sp1_exact=True,
                    gates=[dict(gate=gate, passed=True) for gate in ("sp1_resume", "sp1_transfer")])
        rank = dict(status="completed", qualification=dict(passed=True), cases=[case], unrun_cases=[])
        result = dict(schema="ditserve.workflow.elastic_validation.v1", status="completed",
                      qualification=dict(passed=True), configuration=dict(stage="transfer", seeds=[1], frames=[17]),
                      ranks=[copy.deepcopy(rank), copy.deepcopy(rank)])
        self.assertTrue(self.evidence(result)["qualified"])
        repaired = copy.deepcopy(result)
        repaired["configuration"]["sp_linear_policy"] = "full_rows_except_ffn1"
        # Legacy summary booleans cannot qualify the new actual backend path:
        # it additionally requires unique ranks, real quanta and transitions.
        self.assertFalse(self.evidence(repaired)["qualified"])
        result["configuration"]["seeds"].append(2)
        self.assertFalse(self.evidence(result)["qualified"])
        result["configuration"]["seeds"] = [1]
        result["ranks"][0]["cases"][0]["gates"] = []
        self.assertFalse(self.evidence(result)["qualified"])

    def test_sp_requires_both_seeds_and_all_three_scheduler_prefixes(self):
        modes = [dict(seed=seed, frames=17, mode="baseline", status="qualified",
                      prefixes=[dict(prefix=prefix, passed=True) for prefix in (1, 4, 20)])
                 for seed in (3501, 3502)]
        result = dict(schema="ditserve.workflow.serving.sp.v1", status="completed",
                      qualification=dict(passed=True), configuration=dict(
                          seeds=[3501, 3502], frames=[17], variants=["baseline"]),
                      ranks=[dict(status="completed", variants=copy.deepcopy(modes)) for _ in range(2)])
        self.assertTrue(self.evidence(result)["qualified"])
        result["ranks"][1]["variants"][1]["prefixes"].pop()
        self.assertFalse(self.evidence(result)["qualified"])

    def test_runtime_roles_cannot_promote_unknown_or_other_experiment_schema(self):
        controls = (("sp2_scratch_cost_ablation", "ditserve.workflow.sp2_performance.v1"),
                    ("isolated_shared_decode_performance", "ditserve.workflow.overlap-performance.v1"))
        for role, expected_schema in controls:
            for schema in ("unrecognized.completed.v1", "ditserve.workflow.next-block.v1", None):
                with self.subTest(role=role, schema=schema):
                    result = dict(schema=schema, status="completed", configuration={"fixed": True},
                                  performance_evidence=True, qualification=dict(passed=True))
                    ev = self.evidence(result, metadata={"role": role}, frozen_config={"fixed": True})
                    self.assertTrue(ev["guarded"] and ev["capacity"])
                    self.assertFalse(ev["qualified"])
                    self.assertFalse(report.performance_qualified(ev, {}))
                    self.assertEqual(report.numeric_status(ev), "unverified")

    def test_runtime_roles_bind_exact_frozen_configuration_before_validator(self):
        controls = (("sp2_scratch_cost_ablation", "sp2_performance"),
                    ("isolated_shared_decode_performance", "overlap_performance"))
        for role, module in controls:
            schema = "ditserve.workflow." + ("sp2_performance" if module == "sp2_performance" else "overlap-performance") + ".v1"
            target = "ditserve.evaluation.workflow_" + module + ".validate_result"
            with self.subTest(role=role), patch(target, return_value=dict(passed=True)):
                result = dict(schema=schema, status="completed", configuration={"fixed": True},
                              performance_evidence=True, qualification=dict(passed=True))
                ev = self.evidence(result, metadata={"role": role}, frozen_config={"fixed": True})
                self.assertTrue(ev["qualified"])
                for overrides in ({"frozen_config": {"fixed": False}},
                                  {"frozen_config": {"fixed": True}, "configuration_sha256": "changed"},
                                  {"frozen_config": None}):
                    with self.subTest(overrides=overrides):
                        ev = self.evidence(result, metadata={"role": role}, **overrides)
                        self.assertFalse(ev["qualified"])
                        self.assertFalse(report.performance_qualified(ev, {}))

    def test_completed_sp_and_vae_claims_need_full_raw_evidence(self):
        controls = (("sp2_scratch_cost_ablation", "ditserve.workflow.sp2_performance.v1"),
                    ("isolated_shared_decode_performance", "ditserve.workflow.overlap-performance.v1"))
        for role, schema in controls:
            with self.subTest(role=role):
                result = dict(schema=schema, status="completed", configuration={}, performance_evidence=True,
                              qualification=dict(passed=True, scopes={name: dict(status="passed")
                                  for name in ("numeric", "capacity", "lifecycle")}))
                ev = self.evidence(result, metadata={"role": role}, frozen_config={})
                self.assertTrue(ev["guarded"] and ev["capacity"])
                self.assertFalse(ev["qualified"])
                self.assertFalse(report.performance_qualified(ev, {}))
                self.assertEqual(report.numeric_status(ev), "unverified")

    def test_compact_diagnostic_keeps_differences_but_not_raw_sample_streams(self):
        value = dict(samples=[1, 2], cases=[dict(max_abs=.1, nvml_samples=[3],
                                              memory_margin_gate=dict(status="failed"))])
        compact = report.compact_diagnostic(value)
        self.assertNotIn("samples", compact)
        self.assertNotIn("nvml_samples", compact["cases"][0])
        self.assertEqual(compact["cases"][0]["max_abs"], .1)

    def test_shape_summary_preserves_numeric_steps_flags_and_all_call_counts(self):
        shapes = [dict(module="blocks.0.to_q", action="global_tile", calls=2,
                       tile_intersections=[[0, 0, 1024]], output_storage_offset=0),
                  dict(module="blocks.1.to_q", action="global_tile", calls=4,
                       tile_intersections=[[0, 0, 1024]], output_storage_offset=0),
                  dict(module="blocks.0.to_k", action="skip_nonlocal_sequence", calls=3)]
        audit = dict(variant="global_tile_1024", shapes=shapes, modules=[r["module"] for r in shapes],
                     applied_calls=6, skipped_calls=3, hooks_removed=True,
                     reference_flags=dict(allow_tf32=False), candidate_flags=dict(allow_tf32=False))
        value = dict(threshold=.05, steps=[dict(step=19, passed=False,
                     comparison=dict(latent_after=dict(max_abs=.2, exact=False)))],
                     intervention_audit=audit, bridge_to_original=dict(replay=dict(
                         passed=True, intervention_audit=copy.deepcopy(audit))))
        original = copy.deepcopy(value)
        compact = report.compact_diagnostic(value)
        self.assertEqual(value, original)
        self.assertEqual(compact["steps"], original["steps"])
        self.assertEqual(compact["threshold"], .05)
        output = compact["intervention_audit"]
        self.assertNotIn("shapes", output)
        for key in ("modules", "applied_calls", "skipped_calls", "hooks_removed",
                    "reference_flags", "candidate_flags"):
            self.assertEqual(output[key], audit[key])
        summary = output["shapes_summary"]
        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["calls"], 9)
        self.assertEqual(summary["module_calls"], {"blocks.0.to_q": 2, "blocks.1.to_q": 4, "blocks.0.to_k": 3})
        self.assertEqual(summary["actions"], [dict(action="global_tile", records=2, calls=6),
                                             dict(action="skip_nonlocal_sequence", records=1, calls=3)])
        expected = hashlib.sha256(json.dumps(shapes, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        self.assertEqual(summary["canonical_sha256"], expected)
        self.assertIn("source_result", summary["details"])
        self.assertEqual(compact["bridge_to_original"]["replay"]["intervention_audit"], output)

    def test_sp_duplicate_variants_reference_requires_actual_equality(self):
        variant = dict(mode="global_tile_1024", prefixes=[dict(prefix=20, passed=False,
                           steps=[dict(comparison=dict(max_abs=.123))])])
        value = dict(schema="ditserve.workflow.serving.sp.v1", variants=[copy.deepcopy(variant)],
                     ranks=[dict(rank=0, variants=[copy.deepcopy(variant)]),
                            dict(rank=1, variants=[copy.deepcopy(variant)])])
        original = copy.deepcopy(value)
        compact = report.compact_diagnostic(value)
        self.assertNotIn("variants", compact)
        self.assertEqual(compact["variants_reference"]["json_path"], "$.ranks[0].variants")
        self.assertEqual(compact["ranks"], value["ranks"])
        self.assertEqual(value, original)
        value["variants"][0]["prefixes"][0]["steps"][0]["comparison"]["max_abs"] = .124
        compact = report.compact_diagnostic(value)
        self.assertEqual(compact["variants"], value["variants"])
        self.assertNotIn("variants_reference", compact)
        original["schema"] = "another.schema"
        self.assertIn("variants", report.compact_diagnostic(original))

    def test_report_shape_summary_keeps_original_artifact_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "measurements/diagnostic"
            output.mkdir(parents=True)
            result = dict(status="qualification_failed", intervention_audit=dict(
                variant="full_rows_linear", modules=["proj_out"], applied_calls=2, skipped_calls=0,
                hooks_removed=True, shapes=[dict(module="proj_out", action="pad_full_rows", calls=2)]))
            source = output / "run.json"
            source.write_text(json.dumps(result))
            original = source.read_bytes()
            job = dict(id="diagnostic", family="sp", output=str(output), devices=[0, 1])
            (root / "study.json").write_text(json.dumps(dict(root=str(root), jobs=[job], selections={})))
            record = report.build_report(root)
            diagnostic = record["metrics"]["diagnostics"][0]
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(diagnostic["source_result"]["path"], str(source))
            self.assertEqual(diagnostic["source_result"]["sha256"], hashlib.sha256(original).hexdigest())
            self.assertIn("shapes_summary", diagnostic["evidence"]["intervention_audit"])
            self.assertEqual(record["artifacts"][0]["sha256"], hashlib.sha256(original).hexdigest())

    def test_standalone_report_keeps_input_values_and_screening_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, out = root / "input.json", root / "measurements/job"
            out.mkdir(parents=True)
            config = dict(arms=[dict(id="c8-d1", cpu_threads=8)])
            input_path.write_text(json.dumps(config))
            result = dict(schema="ditserve.workflow.next-block.v1", status="completed",
                          arms=[dict(id="c8-d1", result=dict(requests=[row()], metrics=dict(wall_ms=50)))])
            (out / "run.json").write_text(json.dumps(result))
            job = dict(id="job", family="cpu", input=str(input_path),
                       configuration_sha256=report.digest(config), output=str(out), devices=[0])
            (root / "study.json").write_text(json.dumps(dict(root=str(root), jobs=[job], selections={})))
            record = report.build_report(root, final=True)
            self.assertEqual(record["config"]["inputs"]["job"]["configuration"], config)
            self.assertTrue(record["config"]["inputs"]["job"]["matches_frozen_digest"])
            self.assertEqual(record["requests"][0]["observation_id"], "job/c8-d1")
            self.assertEqual(record["requests"][0]["outputs"], row()["outputs"])


if __name__ == "__main__":
    unittest.main()
