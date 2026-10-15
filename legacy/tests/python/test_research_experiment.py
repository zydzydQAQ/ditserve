"""Synthetic executor fixtures for bounded scheduling, never GPU measurements."""

import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from ditserve.benchmark.common import canonical_hash, seal
from ditserve.research.budget import BudgetStop
from ditserve.research.common import atomic_json, read_json
from ditserve.research.evidence import trajectory_evidence
from ditserve.research.experiment import Experiment, IneligibleAction, run_state_mechanism
from ditserve.research.matrix import ACTIONS, RECIPES


class FakeExperiment:
    def __init__(self, root, *, bad_dev=False, stop_after=None):
        self.root = Path(root)
        self.config = dict(schema_version="ditserve.research.protocol.v1", model_id="fixture", revision="rev", steps=32,
            dev_seeds=[101, 102], holdout_seeds=[201, 202], recipes=copy.deepcopy(list(RECIPES)),
            transition_step=16, max_joint_candidates=2, epsilon=.05, actions=list(ACTIONS), suffix_thresholds=[.04, 0.],
            prompts=[dict(id=name, split="dev", text=name) for name in ("kettle", "pendulum", "pinwheel", "origami")] +
                    [dict(id=name, split="holdout", text=name) for name in ("tram", "aquarium")])
        self.frozen = dict(sha256="f" * 64, source_sha256="a" * 64)
        self.counter = self.checks = 0
        self.worker = SimpleNamespace(torch=SimpleNamespace(OutOfMemoryError=MemoryError))
        self.calls, self.records = [], []
        self.bad_dev, self.stop_after = bad_dev, stop_after
        shape = dict(height=480, width=832, frames=17, steps=32)
        atomic_json(self.root / "probe.json", dict(selected_shape=shape, completed_shapes=[shape, dict(shape, frames=33)]))
        atomic_json(self.root / "correctness.json", dict(status="passed", exact_refresh="passed", protocol_sha256="f" * 64))
        atomic_json(self.root / "calibration.json", seal(dict(status="selected", selected=dict(recipe_id="t008"), protocol_sha256="f" * 64)))

    def check_unit(self):
        self.checks += 1
        if self.stop_after is not None and self.checks > self.stop_after:
            raise BudgetStop("fixture budget exhausted")

    def spec(self, prompt, seed, shape, recipe):
        self.counter += 1
        return dict(shape, **{k: recipe[k] for k in ("cache", "threshold", "max_continuous")},
            request_id="fixture-%d" % self.counter, prompt=prompt["text"], negative_prompt="", seed=seed, guidance_scale=5.)

    def identity(self, spec, prefix, action="none", suffix=None, at_step=16):
        config = lambda value: {k: value[k] for k in ("cache", "threshold", "max_continuous")}
        return dict(model_id="fixture", revision="rev", backend=dict(attention="flash", gpu_name="fixture"),
            precision=dict(transformer="float16", vae="float32"),
            input={k: spec[k] for k in ("prompt", "negative_prompt", "seed")},
            spec={k: spec[k] for k in ("height", "width", "frames", "steps", "guidance_scale")},
            prefix=config(prefix), suffix=config(suffix or prefix), action=dict(kind=action, at_step=0 if action == "none" else at_step),
            layout_history=[dict(at_step=0, ranks=[0], epoch=0)], protocol_sha256="f" * 64, source_sha256="a" * 64)

    def generate(self, spec, *, action="none", suffix=None, at_step=16, urgent_spec=None, admission=None):
        self.calls.append(dict(prompt=spec["prompt"], seed=spec["seed"], action=action, frames=spec["frames"], suffix=suffix))
        if action == "exact_refresh":
            raise IneligibleAction("fixture: certificate unavailable at step 16")
        offset = 0 if suffix is None else {.08: 0., .04: 1., 0.: 2.}[suffix["threshold"]]
        cost = {"none": 100., "keep": 100., "hot": 80., "cold": 90.}[action] + offset
        if admission is not None:
            admission.update(main_completed=True, urgent_decode_completed=True)
        return dict(status="completed", critical_path_ms=cost, elapsed_ms=cost + 20., output_sha256="b" * 64,
            bad_quality=self.bad_dev and spec["prompt"] == "origami" and spec["seed"] == 102 and
                        suffix is not None and suffix["threshold"] == .04,
            urgent=None if urgent_spec is None else dict(status="completed", output_sha256="c" * 64, bad_quality=False))

    @staticmethod
    def serializable(run):
        return {k: v for k, v in run.items() if k != "urgent"}

    def quality(self, run, reference, identity, reference_identity):
        loss = .2 if run.get("bad_quality") else .02
        frames = identity["spec"]["frames"]
        metrics = dict(status="ok", finite=True, range_valid=True, lpips_status="ok", lpips_mean=loss,
            lpips_p95=loss, lpips_per_frame=[loss] * frames, temporal_delta_rmse=.01, frame_count=frames)
        return trajectory_evidence(identity, metrics, correctness="passed", completed_steps=32,
            reference_identity=reference_identity, reference_output_sha256=reference["output_sha256"],
            candidate_output_sha256=run["output_sha256"])

    def record(self, row):
        result = seal(row)
        self.records.append(result)
        return result


class BoundedMechanismTests(unittest.TestCase):
    def test_both_frozen_texts_are_ready_before_measured_parent_prepare(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as directory:
            exp = object.__new__(Experiment)
            exp.root, exp.prompt_bank = Path(directory), None
            warmed = []
            exp.text_identity = lambda spec: warmed.append(spec["request_id"]) or {"mode": "fixture"}
            state = SimpleNamespace(step_index=0, latents=torch.zeros(1), events=[], prepare_ms=0.)
            def prepare(spec):
                self.assertEqual(warmed, ["main", "urgent"])
                return state
            exp.adapter = SimpleNamespace(prepare=prepare, release=lambda value: None,
                finish=lambda value: np.zeros((1, 1, 2, 2, 3), dtype=np.float32))
            run = exp.generate(dict(request_id="main", steps=0), urgent_spec=dict(request_id="urgent"))
            self.assertEqual(run["status"], "completed")

    def test_quality_cells_once_then_small_separate_counterbalanced_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = FakeExperiment(directory)
            result = run_state_mechanism(exp)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["selected_candidates"]), 2)
            self.assertEqual(len(result["development"]), 16)
            self.assertEqual(len(result["holdout"]), 8)
            self.assertEqual(len(result["state_trials"]), 8)  # 2 candidates x 2 conditions x 2 orders.
            self.assertTrue(all(row["status"] == "completed" for row in result["development"]))
            self.assertFalse(result["holdout_used_for_selection"])
            selection = read_json(Path(directory) / "action-selection.json")
            self.assertEqual(len(selection["ineligible"]), 3)
            self.assertEqual(result["gates"]["memory_admission"]["status"], "insufficient_evidence")
            self.assertEqual(len(result["memory_trials"]), 1)
            self.assertEqual(result["memory_trials"][0]["status"], "no_admission_difference")
            for trial in result["state_trials"]:
                self.assertEqual(trial["arms"]["candidate"]["cost_scope"], "full_critical_path")
                self.assertIn("urgent", trial["arms"]["candidate"])

    def test_two_candidates_at_one_shape_do_not_complete_a_crossover_design(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = FakeExperiment(directory)
            probe = read_json(exp.root / "probe.json")
            shape = probe["selected_shape"]
            probe.update(completed_shapes=[shape], cost_plan=dict(
                paired_conditions=[shape], candidate_limit=2))
            atomic_json(exp.root / "probe.json", probe)
            result = run_state_mechanism(exp)
            self.assertEqual(len(result["selected_candidates"]), 2)
            self.assertEqual(result["paired_conditions"], [shape])
            self.assertEqual(len(result["state_trials"]), 4)
            gate = result["gates"]["state_crossover"]
            self.assertEqual(len({c["comparison_sha256"] for c in gate["conditions"]}), 2)
            self.assertEqual(gate["status"], "insufficient_evidence")
            self.assertFalse(gate["supported"])

    def test_unqualified_development_candidate_never_reaches_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = FakeExperiment(directory, bad_dev=True)
            result = run_state_mechanism(exp)
            self.assertEqual(len(result["selected_candidates"]), 2)  # Frozen shortlist never replaced.
            self.assertEqual(len(result["development"]), 16)
            self.assertEqual(len(result["development_qualified_candidates"]), 1)
            self.assertEqual(len(result["holdout"]), 4)
            self.assertEqual({row["candidate_id"] for row in result["holdout"]}, set(result["development_qualified_candidates"]))

    def test_partial_budget_keeps_missing_evidence_and_does_not_select(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = FakeExperiment(directory, stop_after=2)
            with self.assertRaises(BudgetStop):
                run_state_mechanism(exp)
            partial = read_json(Path(directory) / "mechanism.json")
            self.assertEqual(len(partial["screen"]), 2)
            self.assertEqual(partial["selected_candidates"], [])
            self.assertEqual(partial["gates"]["state_crossover"]["status"], "insufficient_evidence")
            self.assertFalse((Path(directory) / "action-selection.json").exists())

    def test_failed_or_different_protocol_correctness_blocks_all_generation(self):
        for record in (dict(status="failed", protocol_sha256="f" * 64),
                       dict(status="passed", protocol_sha256="e" * 64)):
            with tempfile.TemporaryDirectory() as directory:
                exp = FakeExperiment(directory)
                atomic_json(exp.root / "correctness.json", record)
                with self.assertRaises(IneligibleAction):
                    run_state_mechanism(exp)
                self.assertEqual(exp.calls, [])


if __name__ == "__main__":
    unittest.main()
