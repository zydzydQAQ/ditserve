"""Bounded planning fixtures; no model loading or measured performance claims."""

import copy
import subprocess
import sys
import unittest

from ditserve.benchmark.common import canonical_hash, seal
from ditserve.research.evidence import trajectory_evidence
from ditserve.research.matrix import (RECIPES, build_matrix, materialize_joint_jobs,
                                      select_initial_recipe, select_joint_candidates)


def protocol():
    return dict(schema_version="ditserve.research.protocol.v1", model_id="fixture", revision="rev", steps=32,
        dev_seeds=[101, 102], holdout_seeds=[201, 202], recipes=copy.deepcopy(list(RECIPES)),
        transition_step=16, max_joint_candidates=2, epsilon=.05,
        prompts=[dict(id=name, split="dev", text=name + " prompt") for name in ("kettle", "pendulum", "pinwheel", "origami")] +
                [dict(id=name, split="holdout", text=name + " prompt") for name in ("tram", "aquarium")])


SHAPE = dict(height=480, width=832, frames=17, steps=32)


def initial_records(config, *, bad_quality=None):
    matrix = build_matrix(config, SHAPE)
    records = []
    for job in matrix["initial_jobs"]:
        recipe_id = job["recipe_id"]
        loss = .1 if recipe_id == bad_quality else .02 if recipe_id != "off" else 0.
        identity = dict(model_id=config["model_id"], revision=config["revision"],
            backend=dict(attention="flash", gpu_type="fixture"), precision=dict(transformer="float16", vae="float32"),
            input=dict(prompt=job["prompt"], negative_prompt="", seed=job["seed"]),
            spec=dict(SHAPE, guidance_scale=5.), prefix=job["prefix"], suffix=job["suffix"], action=job["action"],
            layout_history=[dict(at_step=0, layout="sp1")], protocol_sha256=canonical_hash(config), source_sha256="a" * 64)
        reference = copy.deepcopy(identity)
        reference.update(prefix=dict(cache=False, threshold=0., max_continuous=1), suffix=dict(cache=False, threshold=0., max_continuous=1))
        metrics = dict(status="ok", finite=True, range_valid=True, lpips_status="ok", lpips_mean=loss,
            lpips_p95=loss, lpips_per_frame=[loss] * 17, temporal_delta_rmse=.01, frame_count=17)
        proof = trajectory_evidence(identity, metrics, correctness="passed", completed_steps=32,
            reference_identity=reference, reference_output_sha256="b" * 64, candidate_output_sha256="c" * 64)
        records.append(dict(recipe_id=recipe_id, prompt_id=job["prompt_id"], seed=job["seed"], split="dev",
            status="completed", measurement_kind="measured_gpu", evidence=proof,
            phase_cost_ms=dict(prepare=10. if recipe_id == "off" else 9999.,
                denoise={"off": 100., "t004": 80., "t008": 60.}[recipe_id], decode=10.)))
    return records


def screen_records(matrix):
    return [dict(candidate_id=job["candidate_id"], prompt_id=job["prompt_id"], seed=job["seed"], split="dev",
        status="completed", measurement_kind="measured_gpu", correctness_status="passed", cost_ms=100. + index,
        context_sha256="a" * 64, protocol_sha256=matrix["protocol_sha256"])
        for index, job in enumerate(matrix["screen_jobs"])]


class ResearchMatrixTests(unittest.TestCase):
    def test_import_and_planning_do_not_load_gpu_or_numpy(self):
        completed = subprocess.run([sys.executable, "-c",
            "import sys; from ditserve.research import matrix; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_bounded_matrix_deduplicates_suffix_and_defers_joint_and_holdout(self):
        matrix = build_matrix(protocol())
        self.assertEqual(len(matrix["initial_jobs"]), 24)
        self.assertEqual(len(matrix["screen_candidates"]), 24)
        self.assertEqual(len(matrix["screen_jobs"]), 24)
        self.assertTrue(matrix["requires_selected_shape"])
        self.assertFalse(matrix["executes_gpu_work"])
        self.assertEqual(matrix["joint_validation_jobs"], [])
        self.assertEqual(matrix["holdout_jobs"], [])
        for row in matrix["screen_candidates"]:
            self.assertEqual(row["action"]["at_step"], 16)
            self.assertLessEqual(row["policy_changes"], 1)
            self.assertLessEqual(row["suffix"]["threshold"], row["prefix"]["threshold"])
        self.assertEqual({row["seed"] for row in matrix["holdout_inputs"]}, {201, 202})

    def test_initial_selection_uses_all_dev_quality_and_excludes_cpu_variability(self):
        config = protocol()
        result = select_initial_recipe(initial_records(config), config, SHAPE)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected"]["recipe_id"], "t008")
        self.assertEqual(result["selected"]["shared_prepare_ms"], 10.)
        self.assertEqual(result["selected"]["comparable_service_ms"], 80.)
        result = select_initial_recipe(initial_records(config, bad_quality="t008"), config)
        self.assertEqual(result["selected"]["recipe_id"], "t004")

    def test_missing_unknown_or_duplicate_calibration_cannot_bias_selection(self):
        config = protocol()
        for change in ("missing", "duplicate", "unknown"):
            records = initial_records(config)
            if change == "missing":
                records.pop()
            elif change == "duplicate":
                records[-1] = copy.deepcopy(records[-2])
            else:
                proof = records[-1]["evidence"]
                proof["metrics"]["status"] = "unknown"
                records[-1]["evidence"] = seal(proof)
            self.assertEqual(select_initial_recipe(records, config)["status"], "insufficient_evidence")

    def test_cross_recipe_embedding_and_reference_mismatch_blocks_selection(self):
        config = protocol()
        records = initial_records(config)
        for row in records:
            proof = row["evidence"]
            encoding = dict(mode="frozen_cpu_embeddings_mechanism_only", hash=row["recipe_id"])
            proof["identity"]["input"]["text_encoding"] = encoding
            proof["reference_identity"]["input"]["text_encoding"] = encoding
            proof["trajectory_sha256"] = canonical_hash(proof["identity"])
            proof["reference_trajectory_sha256"] = canonical_hash(proof["reference_identity"])
            row["evidence"] = seal(proof)
        result = select_initial_recipe(records, config)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["reason"], "cross_recipe_input_or_reference_mismatch")

    def test_mixing_shapes_backends_sources_or_layouts_cannot_fill_eight_cells(self):
        config = protocol()
        for field, value in (("spec", dict(SHAPE, frames=33, guidance_scale=5.)),
                             ("backend", dict(attention="math")), ("source_sha256", "d" * 64),
                             ("layout_history", [dict(at_step=0, layout="sp2")])):
            records = initial_records(config)
            proof = records[-1]["evidence"]
            proof["identity"][field] = value
            proof["trajectory_sha256"] = canonical_hash(proof["identity"])
            records[-1]["evidence"] = seal(proof)
            with self.subTest(field=field):
                self.assertEqual(select_initial_recipe(records, config)["status"], "insufficient_evidence")
        self.assertEqual(select_initial_recipe(initial_records(config), config, dict(SHAPE, frames=33))["status"], "insufficient_evidence")

    def test_holdout_never_enters_initial_or_joint_selection(self):
        config = protocol()
        records = initial_records(config)
        records[0].update(prompt_id="tram", split="holdout", seed=201)
        with self.assertRaises(ValueError):
            select_initial_recipe(records, config)
        matrix = build_matrix(config)
        records = screen_records(matrix)
        records[0].update(prompt_id="aquarium", split="dev", seed=101)
        with self.assertRaises(ValueError):
            select_joint_candidates(records, matrix=matrix)

    def test_cost_screen_selects_max_two_and_only_then_expands_validation(self):
        matrix = build_matrix(protocol(), SHAPE)
        result = select_joint_candidates(screen_records(matrix), matrix=matrix)
        self.assertEqual(result["status"], "selected_for_validation")
        self.assertEqual(len(result["selected"]), 2)
        self.assertFalse(result["quality_qualified"])
        dev_jobs = materialize_joint_jobs(matrix, result)
        holdout_jobs = materialize_joint_jobs(matrix, result, split="holdout")
        self.assertEqual(len(dev_jobs), 16)
        self.assertEqual(len(holdout_jobs), 8)
        self.assertTrue(all(row["requires_development_quality"] for row in holdout_jobs))
        self.assertFalse(any(row["selection_permitted"] for row in holdout_jobs))
        with self.assertRaises(ValueError):
            select_joint_candidates(screen_records(matrix), matrix=matrix, limit=3)

    def test_missing_cost_or_mixed_context_is_insufficient_not_slow(self):
        matrix = build_matrix(protocol())
        for change in ("missing", "failed", "context"):
            records = screen_records(matrix)
            if change == "missing":
                records.pop()
            elif change == "failed":
                records[-1]["status"] = "failed"
            else:
                records[-1]["context_sha256"] = "b" * 64
            result = select_joint_candidates(records, matrix=matrix)
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertEqual(result["selected"], [])

    def test_joint_jobs_reject_changed_selection_or_another_matrix(self):
        matrix = build_matrix(protocol(), SHAPE)
        selection = select_joint_candidates(screen_records(matrix), matrix=matrix)
        altered = copy.deepcopy(selection)
        altered["selected"][0]["candidate"]["action"]["at_step"] = 24
        with self.assertRaises(ValueError):
            materialize_joint_jobs(matrix, seal(altered))
        with self.assertRaises(ValueError):
            materialize_joint_jobs(build_matrix(protocol(), dict(SHAPE, frames=33)), selection)

    def test_frozen_wrapper_hash_and_selected_prefix_bind_reduced_screen(self):
        matrix = build_matrix(protocol(), SHAPE, protocol_sha256="f" * 64,
                              initial_recipe_id="t004", allowed_actions=["keep", "hot", "cold"])
        self.assertEqual(len(matrix["initial_jobs"]), 24)
        self.assertEqual(len(matrix["screen_candidates"]), 6)
        self.assertEqual(matrix["protocol_sha256"], "f" * 64)
        self.assertEqual({row["prefix_recipe_id"] for row in matrix["screen_candidates"]}, {"t004"})

    def test_certified_ineligibility_excludes_action_but_missing_is_unknown(self):
        matrix = build_matrix(protocol(), SHAPE, initial_recipe_id="t004")
        rows = screen_records(matrix)
        rows[0].update(status="ineligible", reason="exact certificate unavailable at frozen boundary")
        del rows[0]["cost_ms"]
        selected = select_joint_candidates(rows, matrix=matrix)
        self.assertEqual(selected["status"], "selected_for_validation")
        self.assertEqual(len(selected["ineligible"]), 1)
        self.assertNotIn(rows[0]["candidate_id"], [r["candidate_id"] for r in selected["selected"]])


if __name__ == "__main__":
    unittest.main()
