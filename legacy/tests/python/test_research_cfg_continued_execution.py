import contextlib
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ditserve.research.common import atomic_json, read_json
from ditserve.research.cfg_cost_plan import continuation_cost_plan
from ditserve.research.cfg_experiment import CfgExperiment
from ditserve.research.cfg_protocol import DEFAULT_CONFIG
from ditserve.research.dual_matrix import seal


class ContinuedExecutionTests(unittest.TestCase):
    def experiment(self, divergent=False, raises=False):
        exp = object.__new__(CfgExperiment)
        exp.config = read_json(DEFAULT_CONFIG)
        exp.shape, exp.backends = exp.config["shapes"]["main"], exp.config["attention"]
        exp.costs, exp.implementations = dict(a100=6.), {"paired_008": "split_prefix"}
        exp.unit = lambda *a: contextlib.nullcontext()
        exp.records, exp.calls = [], []
        exp.record = lambda kind, case, **kw: exp.records.append(dict(type=kind, **kw))
        exp.write = lambda name, value: value

        def generate(item, recipe, **kw):
            impl = exp.implementations.get(recipe, "native")
            exp.calls.append((recipe, impl, kw.get("action", "stay")))
            if raises and impl == "oracle":
                raise RuntimeError("oracle fault")
            return dict(latent_sha256="latent", output_sha256="video", elapsed_ms=6000,
                        traces=[dict(step=i, state_sha256="different" if divergent and impl == "oracle" else "same")
                                for i in range(1, 33)])
        exp.generate = generate
        return exp

    def test_current_oracle_is_generated_and_stepwise_state_is_checked(self):
        exp = self.experiment()
        self.assertEqual(exp.correctness("a100")["status"], "completed")
        paired = [c for c in exp.calls if c[0] == "paired_008"]
        self.assertEqual(paired, [("paired_008", "split_prefix", "stay"),
                                 ("paired_008", "split_prefix", "local_full_migrate"),
                                 ("paired_008", "oracle", "stay")])
        evidence = next(r for r in exp.records if r["type"] == "cfg_implementation_equivalence")
        self.assertTrue(evidence["equal"])
        self.assertFalse(evidence["timing_eligible"])
        with self.assertRaisesRegex(RuntimeError, "differs"):
            self.experiment(divergent=True).correctness("a100")

    def test_oracle_failure_restores_execution_identity(self):
        exp = self.experiment(raises=True)
        with self.assertRaisesRegex(RuntimeError, "oracle fault"):
            exp.correctness("a100")
        self.assertEqual(exp.implementations["paired_008"], "split_prefix")

    def refinement(self, **kwargs):
        exp = self.experiment(**kwargs)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        exp.root = Path(temp.name)
        exp.frozen = dict(implementation_refinement={"bound": True})
        atomic_json(exp.root / "execution-plan.json", dict(per_device={"a100":
            dict(correctness_unit_estimate_wall_seconds=40.)}))
        original, exp.arguments = exp.generate, []
        def generate(item, recipe, **options):
            exp.arguments.append(options)
            return original(item, recipe, **options)
        exp.generate = generate
        return exp

    def test_refinement_three_new_trajectories_cover_shadow_restore_and_oracle(self):
        exp = self.refinement()
        summary = exp.correctness("a100")
        self.assertEqual(summary["validation_mode"], "refinement_three_trajectories")
        self.assertEqual(exp.calls, [("paired_008", "split_prefix", "stay"),
            ("paired_008", "split_prefix", "local_full_migrate"), ("paired_008", "oracle", "stay")])
        self.assertTrue(exp.arguments[0]["diagnostic"])
        self.assertFalse(exp.arguments[0]["diagnostic_reference"])
        self.assertFalse(exp.arguments[1].get("diagnostic", False))
        self.assertEqual({r["type"] for r in exp.records}, {"cfg_correctness",
            "cfg_implementation_equivalence", "cfg_diagnostic_neutrality"})
        self.assertTrue(all(r["equal"] and not r["timing_eligible"] for r in exp.records))
        self.assertTrue(summary["diagnostic_gpu_validated"])

    def test_refinement_rejects_divergence_and_restores_identity_on_failure(self):
        for options in (dict(divergent=True), dict(raises=True)):
            with self.subTest(options=options):
                exp = self.refinement(**options)
                with self.assertRaises(RuntimeError):
                    exp.correctness("a100")
                self.assertEqual(exp.implementations["paired_008"], "split_prefix")

    def test_refinement_requires_all_32_step_states(self):
        exp = self.refinement()
        original = exp.generate
        def short(*args, **kwargs):
            run = original(*args, **kwargs)
            run["traces"] = run["traces"][:31]
            return run
        exp.generate = short
        with self.assertRaisesRegex(RuntimeError, "differs"):
            exp.correctness("a100")


class ContinuedCostPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "units").mkdir()
        atomic_json(root / "protocol.json", dict(sha256="parent"))
        self.frozen = dict(sha256="new", source_sha256="source", prototype_continuation=dict(parent_run=str(root)))
        self.ledger = dict(active=None, used_gpu_seconds=1000, stage_used_gpu_seconds={}, sessions=[],
            reallocations=[dict(stage_allowances_gpu_seconds=dict(correctness=1000, calibration=5000, capacity=900),
                                baseline_stage_used={})])
        for role in ("a100", "v100"):
            atomic_json(root / ("screen-" + role + "-job.json"), dict(physical_devices=[dict(uuid="GPU-" + role)]))
            for name in ("screen-" + role + "-stage_a-pair", "correctness-" + role + "-paired_008"):
                atomic_json(root / "units" / (name + ".json"), seal(dict(unit_id=name, started_at=1., ended_at=11.,
                    status="completed", performance_valid=True, protocol_sha256="parent")))
            for stage in ("correctness", "calibration"):
                self.ledger["sessions"].append(dict(stage=stage, protocol_sha256="parent", devices=["GPU-" + role],
                                                    status="completed", wall_seconds=15.))
        self.mock = patch("ditserve.research.cfg_continuation.screening_decisions", return_value={
            "independent_004": dict(eligible=True), "paired_008": dict(eligible=True)})
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_plan_contains_full_development_and_holdout_and_setup_reserve(self):
        plan = continuation_cost_plan(self.frozen, self.ledger)
        self.assertEqual(plan["status"], "fits")
        self.assertEqual(sum(r["formal_dev_pairs"] for r in plan["per_device"].values()), 64)
        self.assertEqual(sum(r["held_out_pairs"] for r in plan["per_device"].values()), 8)
        self.assertGreater(plan["estimates_gpu_seconds"]["calibration"], 72 * 10 * 1.3)

    def test_capacity_balance_never_covers_correctness_shortfall(self):
        self.ledger["reallocations"][0]["stage_allowances_gpu_seconds"]["correctness"] = 1
        self.assertEqual(continuation_cost_plan(self.frozen, self.ledger)["status"], "budget_stopped")

    def test_active_ledger_rejected_before_planning(self):
        self.ledger["active"] = dict(stage="calibration")
        with self.assertRaisesRegex(ValueError, "settled"):
            continuation_cost_plan(self.frozen, self.ledger)


if __name__ == "__main__":
    unittest.main()
