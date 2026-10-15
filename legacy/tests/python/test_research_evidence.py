"""Pure contract fixtures; constructed measurement labels are never GPU results."""

import copy
import unittest

from ditserve.benchmark.common import seal
from ditserve.research.evidence import (memory_admission_gate, qualify_trajectory, research_gates,
    sp_path_gate, state_crossover_gate, trajectory_evidence, trajectory_fingerprint)


OFF = dict(cache=False, threshold=0., max_continuous=1)


def identity(**updates):
    result = dict(model_id="fixture-model", revision="fixture-revision",
        backend=dict(attention="flash", gpu_type="fixture-only", transformer="wan"),
        precision=dict(transformer="float16", vae="float32", text_encoder="float32"),
        input=dict(prompt="fixture", negative_prompt="", seed=101),
        spec=dict(height=256, width=256, frames=3, steps=32, guidance_scale=5.),
        prefix=dict(OFF), action=dict(kind="none", at_step=0), suffix=dict(OFF),
        layout_history=[dict(at_step=0, layout="sp1")], protocol_sha256="a" * 64, source_sha256="b" * 64)
    result.update(updates)
    return result


def metrics(loss=.02, frames=3):
    return dict(status="ok", finite=True, range_valid=True, lpips_status="ok",
        lpips_mean=loss, lpips_p95=loss, lpips_per_frame=[loss] * frames,
        temporal_delta_rmse=.01, frame_count=frames)


def evidence(ident=None, loss=.02, **updates):
    ident = identity() if ident is None else ident
    reference = copy.deepcopy(ident)
    reference.update(prefix=dict(OFF), suffix=dict(OFF), action=dict(kind="none", at_step=0))
    reference["layout_history"] = [dict(at_step=0, layout="sp1")]
    args = dict(correctness="passed", completed_steps=32, reference_output_sha256="c" * 64,
                candidate_output_sha256="d" * 64, reference_identity=reference)
    args.update(updates)
    return trajectory_evidence(ident, metrics(loss, ident["spec"]["frames"]), **args)


def arm(name, cost, *, phase=None):
    ident = identity(action=dict(kind=name if name in ("keep", "hot") else "none", at_step=16 if name in ("keep", "hot") else 0))
    if phase:
        ident.update(layout_history=[dict(at_step=0, layout=name)],
                     prefix=dict(cache=True, threshold=.08, max_continuous=1),
                     suffix=dict(cache=True, threshold=.08, max_continuous=1))
    result = dict(cost_ms=cost, identity=ident, evidence=evidence(ident), cost_scope="full_critical_path",
                  urgent=dict(identity=identity(), evidence=evidence()))
    if phase:
        result.update(cost_scope="net_layout_window", collectives_included=True, transitions_included=True,
                      observed_path=phase)
    return result


def pair(condition, repeat, costs, *, phase=None):
    names = list(costs)
    return dict(condition_id=condition, pair_id="repeat-%d" % repeat, order=names if repeat % 2 == 0 else names[::-1],
        status="completed", measurement_kind="measured_gpu", phase=phase,
        arms={name: arm(name, cost, phase=phase) for name, cost in costs.items()})


class TrajectoryEvidenceTests(unittest.TestCase):
    def test_exact_complete_trajectory_qualifies_and_exports_all_quality_fields(self):
        result = qualify_trajectory(evidence(), identity())
        self.assertEqual(result["status"], "qualified")
        self.assertEqual(result["lpips_per_frame"], [.02] * 3)
        self.assertEqual(result["temporal_delta_rmse"], .01)
        self.assertEqual(qualify_trajectory(evidence(loss=.051), identity())["status"], "failed")

    def test_no_qualification_reuse_for_any_bound_configuration_dimension(self):
        original = identity()
        proof = evidence(original)
        changes = dict(model_id="other", revision="other", backend={"attention": "math"},
            precision={"transformer": "bfloat16"}, input=dict(prompt="other", negative_prompt="", seed=101),
            spec=dict(original["spec"], frames=5), prefix=dict(cache=True, threshold=.04, max_continuous=1),
            action=dict(kind="hot", at_step=16), suffix=dict(cache=True, threshold=.08, max_continuous=1),
            layout_history=[dict(at_step=0, layout="sp2")], protocol_sha256="e" * 64, source_sha256="f" * 64)
        for key, changed in changes.items():
            candidate = copy.deepcopy(original)
            candidate[key] = changed
            with self.subTest(field=key):
                self.assertNotEqual(trajectory_fingerprint(candidate), trajectory_fingerprint(original))
                result = qualify_trajectory(proof, candidate)
                self.assertFalse(result["qualified"])
                self.assertIn("trajectory_identity_mismatch", result["reasons"])

    def test_full_frames_steps_source_and_raw_output_provenance_are_required(self):
        for change in (dict(completed_steps=16), dict(reference_output_sha256=None),
                       dict(candidate_output_sha256=None), dict(correctness="unknown"),
                       dict(reference_identity=None), dict(measurement_kind="cpu_fixture")):
            with self.subTest(change=change):
                self.assertEqual(qualify_trajectory(evidence(**change), identity())["status"], "insufficient")
        changed = evidence()
        changed["metrics"]["lpips_per_frame"] = [.02]
        self.assertEqual(qualify_trajectory(seal(changed), identity())["status"], "insufficient")

    def test_incompatible_reference_backend_or_seed_never_passes(self):
        for reference in (identity(backend=dict(attention="math")),
                          identity(input=dict(prompt="fixture", negative_prompt="", seed=102))):
            result = qualify_trajectory(evidence(reference_identity=reference), identity())
            self.assertFalse(result["qualified"])
            self.assertIn("reference_input_recipe_or_source_mismatch", result["reasons"])

    def test_embedding_content_is_part_of_input_qualification(self):
        candidate = identity()
        candidate["input"]["text_encoding"] = {"prompt": {"sha256": "a" * 64}}
        reference = copy.deepcopy(candidate)
        reference["input"]["text_encoding"]["prompt"]["sha256"] = "b" * 64
        result = qualify_trajectory(evidence(candidate, reference_identity=reference), candidate)
        self.assertFalse(result["qualified"])
        self.assertIn("reference_input_recipe_or_source_mismatch", result["reasons"])

    def test_no_additive_lpips_or_inconsistent_aggregates(self):
        for key, value in (("aggregation_scope", "prefix_plus_suffix"), ("lpips_mean", .01),
                           ("lpips_p95", .01), ("temporal_delta_rmse", None)):
            proof = evidence()
            proof["metrics"][key] = value
            with self.subTest(key=key):
                self.assertFalse(qualify_trajectory(seal(proof), identity())["qualified"])

    def test_hash_tampering_and_unexpected_metadata_cannot_reuse_qualification(self):
        proof = evidence()
        proof["metrics"]["lpips_mean"] = 0
        self.assertEqual(qualify_trajectory(proof, identity())["reasons"], ["missing_or_changed_evidence"])
        extended = identity(extra_runtime_setting="different")
        self.assertFalse(qualify_trajectory(evidence(), extended)["qualified"])


class ResearchGateTests(unittest.TestCase):
    def test_empty_gates_are_unknown_not_negative_research_results(self):
        result = research_gates()
        for key in ("memory_admission", "state_crossover", "sp_path"):
            self.assertEqual(result[key]["status"], "insufficient_evidence")

    def test_state_requires_repeated_counterbalanced_different_condition_winners(self):
        rows = [pair("roomy", repeat, dict(keep=90., hot=100.)) for repeat in (0, 1)]
        rows += [pair("pressure", repeat, dict(keep=100., hot=90.)) for repeat in (0, 1)]
        self.assertTrue(state_crossover_gate(rows)["supported"])
        self.assertEqual(state_crossover_gate(rows[:-1])["status"], "insufficient_evidence")
        same_order = copy.deepcopy(rows)
        for row in same_order:
            row["order"] = ["keep", "hot"]
        self.assertFalse(state_crossover_gate(same_order)["supported"])

    def test_under_five_percent_is_observed_but_not_a_crossover(self):
        rows = [pair("roomy", repeat, dict(keep=96., hot=100.)) for repeat in (0, 1)]
        rows += [pair("pressure", repeat, dict(keep=100., hot=96.)) for repeat in (0, 1)]
        self.assertEqual(state_crossover_gate(rows)["status"], "not_observed")

    def test_local_restore_cost_or_unqualified_urgent_video_cannot_pass(self):
        rows = [pair("roomy", repeat, dict(keep=90., hot=100.)) for repeat in (0, 1)]
        rows += [pair("pressure", repeat, dict(keep=100., hot=90.)) for repeat in (0, 1)]
        for change in ("cost", "quality"):
            altered = copy.deepcopy(rows)
            if change == "cost":
                for row in altered:
                    row["arms"]["hot"]["cost_scope"] = "local_restore"
            else:
                for row in altered:
                    row["arms"]["hot"]["urgent"]["evidence"] = evidence(loss=.2)
            self.assertFalse(state_crossover_gate(altered)["supported"])

    def test_sp_requires_real_compute_reuse_different_winners_and_net_cost(self):
        rows = [pair("shape17", repeat, dict(sp1=100., sp2=80.), phase="compute") for repeat in (0, 1)]
        rows += [pair("shape17", repeat, dict(sp1=80., sp2=100.), phase="reuse") for repeat in (0, 1)]
        self.assertTrue(sp_path_gate(rows)["supported"])
        for field in ("collectives_included", "transitions_included", "observed_path"):
            altered = copy.deepcopy(rows)
            for row in altered:
                row["arms"]["sp2"][field] = False
            self.assertFalse(sp_path_gate(altered)["supported"])

    def test_memory_requires_real_release_and_execution_of_both_trajectories(self):
        row = dict(trial_id="fixture", status="completed", measurement_kind="measured_gpu",
            keep_rejected=True, keep_rejection_reason="measured_capacity", released_mode="exact_refresh", transfer_complete=True,
            keep_request_identity=identity(),
            before=dict(measurement_kind="measured_gpu", allocated_bytes=1000, reserved_bytes=1100, driver_free_bytes=500),
            after=dict(measurement_kind="measured_gpu", allocated_bytes=500, reserved_bytes=1100, driver_free_bytes=500),
            admitted=dict(allocation_succeeded=True, compute_completed=True, decode_completed=True, identity=identity(), evidence=evidence()),
            resumed=dict(identity=identity(action=dict(kind="exact_refresh", at_step=16)),
                         evidence=evidence(identity(action=dict(kind="exact_refresh", at_step=16)))))
        self.assertTrue(memory_admission_gate([row])["supported"])
        for change in ("execution", "estimate", "no_release", "resume_quality", "artificial", "single_oom", "different_request"):
            changed = copy.deepcopy(row)
            if change == "execution":
                changed["admitted"]["compute_completed"] = False
            elif change == "estimate":
                changed["after"]["measurement_kind"] = "estimate"
            elif change == "no_release":
                changed["after"]["allocated_bytes"] = 1000
            elif change == "resume_quality":
                changed["resumed"]["evidence"] = evidence(loss=.2)
            elif change == "artificial":
                changed["artificially_filled_memory"] = True
            elif change == "single_oom":
                changed["single_request_oom"] = True
            else:
                changed["keep_request_identity"]["input"]["seed"] = 102
            self.assertFalse(memory_admission_gate([changed])["supported"])


if __name__ == "__main__":
    unittest.main()
