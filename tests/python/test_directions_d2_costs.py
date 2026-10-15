"""CPU fakes verify the finite state-cost matrix, evidence and timing boundaries."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.directions import direction2 as d2
from ditserve.directions import direction2_costs as costs


class BudgetStop(RuntimeError):
    pass


class Runtime:
    instances = []
    change_full_output = False
    fail_flush_quality = False
    change_second_keep = False

    def __init__(self, config, settings, check):
        self.check, self.clock, self.states, self.events, self.closed = check, 0., [], [], 0
        self.settings = settings
        self.instances.append(self)
        self.bank = SimpleNamespace(warm=lambda spec: self.tick(.07))
        self.worker = SimpleNamespace(_sync=lambda: None, _memory=self.memory,
            finish=self.finish, release=self.release)
        self.state_adapter = SimpleNamespace(inventory=self.inventory, snapshot=self.snapshot, restore=self.restore)

    def tick(self, seconds):
        self.clock += seconds

    def spec(self, item, frames, policy):
        return dict(item=item, frames=frames, policy=policy)

    def prepare(self, item, frames, policy):
        self.check()
        state = SimpleNamespace(step_index=0, residency="gpu", needs_anchor=False, flushed=False,
            moved=False, contexts={"blocks": SimpleNamespace(buffers={"features": object()})})
        self.states.append(state)
        self.events.append("prepare")
        return state

    def step(self, state, policy):
        self.check()
        assert state.residency == "gpu"
        if state.needs_anchor:
            assert not state.contexts["blocks"].buffers
            state.flushed = True
            self.tick(.03)
        else:
            self.tick(.01)
        state.contexts["blocks"].buffers = {"features": object()}
        state.needs_anchor = False
        state.step_index += 1

    def snapshot(self, state, mode):
        assert mode == "hot" and state.step_index == 16
        self.events.append("snapshot")
        self.tick(.04)
        state.residency, state.moved = "cpu", True
        return dict(transfer_bytes=30, ms=40.)

    def restore(self, state):
        assert state.residency == "cpu"
        self.events.append("restore")
        self.tick(.06)
        state.residency = "gpu"
        return dict(transfer_bytes=30, ms=60.)

    def finish(self, state):
        assert state.step_index == 32
        self.tick(.05)
        value = .2 if state.flushed and self.fail_flush_quality else .02 if state.moved and self.change_full_output else .01
        if self.change_second_keep and len(self.states) == 6:
            value = .015
        return np.full((1, 81, 1, 1, 3), value, dtype=np.float32)

    def release(self, state):
        self.events.append("release")
        state.residency = "released"

    def memory(self):
        return dict(allocated_bytes=1000 + sum(self.inventory(s)["cuda_bytes"] for s in self.states))

    def inventory(self, state):
        cuda = state.residency == "gpu"
        feature = 20 if state.contexts["blocks"].buffers and state.residency != "released" else 0
        return dict(necessary_storage_bytes=10, feature_only_bytes=feature,
                    feature_only_cuda_bytes=feature if cuda else 0,
                    cuda_bytes=10 + feature if cuda else 0, storage_bytes=10 + feature)

    def generate(self, item, frames, policy):
        self.check()
        assert policy == "off" and frames == 81
        self.events.append("reference")
        self.tick(.4)
        return dict(output=np.zeros((1, 81, 1, 1, 3), dtype=np.float32), generation_ms=400.)

    def quality(self, reference, output):
        self.events.append("quality")
        value = float(output.max())
        return dict(status="ok", finite=True, range_valid=True, frame_count=81,
            shape=[1, 81, 480, 832, 3], lpips_status="ok", lpips_mean=value, lpips_per_frame=[value] * 81)

    def close(self):
        self.closed += 1
        for state in self.states:
            state.residency = "released"


class StateCostsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "direction2"
        self.output.mkdir()
        self.config = dict(models=dict(wan=dict(id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            revision="0fad780a534b6463e45facd96134c9f345acfa5b", path="/offline/model")),
            device="cuda:0", direction2={})
        identity = d2._identity(self.config, d2._settings(self.config))
        self.qualified = dict(identity=identity, status="passed", eligible=True, selected_frames=81,
                              cache_quality_qualified=True, cache_performance_qualified=True, returncode=0)
        for name in ("qualify.json", "supervisor-qualify.json"):
            d2._write(self.output / name, self.qualified)
        Runtime.instances = []
        Runtime.change_full_output = Runtime.fail_flush_quality = False
        Runtime.change_second_keep = False
        factory = patch.object(d2, "_runtime_factory", Runtime)
        clock = patch.object(costs.time, "perf_counter", side_effect=lambda: Runtime.instances[-1].clock if Runtime.instances else 0.)
        factory.start(); clock.start()
        self.addCleanup(factory.stop); self.addCleanup(clock.stop)

    def run_costs(self, check=lambda: None):
        return costs.run_phase(self.config, self.output, check, "state_cost")

    def test_fixed_six_trials_direct_transfer_and_correct_cost_comparisons(self):
        report = self.run_costs()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["eligible"] and report["completed_cost_matrix"])
        self.assertTrue(report["repeated_keep_exact"])
        self.assertTrue(report["descriptive_only"])
        for key in ("capacity_qualified", "state_mechanism_qualified", "sp_resize_qualified"):
            self.assertFalse(report[key])
        self.assertEqual([r["action"] for r in report["units"]], ["keep", "full_cpu", "flush", "flush", "full_cpu", "keep"])
        self.assertEqual(Runtime.instances[0].events.count("reference"), 1)
        for row in report["units"]:
            self.assertTrue(row["complete"] and row["quality_passed"])
            self.assertEqual(row["last_completed_step"], 32)
            self.assertEqual(row["item"]["prompt_id"], "origami")
            self.assertEqual(row["item"]["seed"], 101)
            self.assertEqual(row["memory"]["after_release"]["allocated_bytes"], 1000)
            self.assertAlmostEqual(row["timings"]["prefix_ms"], 160.)
            self.assertAlmostEqual(row["timings"]["encoding_setup_ms"], 70.)
            if row["action"] == "full_cpu":
                self.assertEqual(row["transfer"]["d2h_bytes"], 30)
                self.assertEqual(row["transfer"]["h2d_bytes"], 30)
                self.assertEqual(row["state_inventory"]["after_action"]["cuda_bytes"], 0)
                self.assertAlmostEqual(row["timings"]["d2h_ms"], 40.)
                self.assertAlmostEqual(row["timings"]["h2d_ms"], 60.)
                self.assertAlmostEqual(row["timings"]["post_pause_work_ms"], 310.)
                self.assertTrue(row["exact_to_keep"])
            elif row["action"] == "flush":
                self.assertEqual(row["state_inventory"]["after_action"]["feature_only_bytes"], 0)
                self.assertAlmostEqual(row["timings"]["next_step_ms"], 30.)
                self.assertAlmostEqual(row["comparison"]["next_step_over_keep_ms"], 20.)
            else:
                self.assertAlmostEqual(row["timings"]["suffix_complete_ms"], 210.)
        self.assertEqual(Runtime.instances[0].closed, 1)
        self.assertEqual(len(list(self.output.rglob("*.npy"))), 7)

    def test_full_cpu_must_match_keep_exactly_and_failures_remain(self):
        Runtime.change_full_output = True
        report = self.run_costs()
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["eligible"])
        self.assertTrue(report["completed_cost_matrix"])
        failures = [r for r in report["units"] if not r["quality_passed"]]
        self.assertEqual([r["action"] for r in failures], ["full_cpu", "full_cpu"])
        for row in failures:
            self.assertLess(row["quality"]["lpips_mean"], .05)
            self.assertFalse(row["exact_to_keep"])
            self.assertTrue(Path(row["retention"]["files"][0]["path"]).is_file())

    def test_flush_quality_gate_and_reverse_order_are_not_dropped(self):
        Runtime.fail_flush_quality = True
        report = self.run_costs()
        self.assertEqual(len(report["units"]), 6)
        self.assertEqual([r["action"] for r in report["units"] if r["status"] == "failed"], ["flush", "flush"])
        self.assertFalse(report["eligible"])

    def test_repeated_keep_mismatch_blocks_admission_without_dropping_matrix(self):
        Runtime.change_second_keep = True
        report = self.run_costs()
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["eligible"] or report["repeated_keep_exact"])
        self.assertTrue(report["completed_cost_matrix"])
        self.assertEqual(len(report["units"]), 6)
        self.assertIn("repeated keep output mismatch", report["reason"])

    def test_budget_between_offload_and_restore_retains_real_partial_fields(self):
        def check():
            if Runtime.instances and Runtime.instances[0].events[-1:] == ["snapshot"]:
                raise BudgetStop("stop before restore")
        report = self.run_costs(check)
        self.assertEqual(report["status"], "stopped")
        self.assertTrue(report["budget_stopped"])
        self.assertFalse(report["completed_cost_matrix"])
        row = report["units"][-1]
        self.assertEqual(row["stage"], "restore")
        self.assertAlmostEqual(row["timings"]["d2h_ms"], 40.)
        self.assertIsNone(row["timings"]["h2d_ms"])
        self.assertIsNone(row["transfer"]["h2d_bytes"])
        self.assertTrue(Path(report["reference_off"]["retention"]["files"][0]["path"]).exists())
        self.assertTrue(Path(report["units"][0]["retention"]["files"][0]["path"]).exists())
        self.assertEqual(Runtime.instances[0].closed, 1)

    def test_budget_after_quality_preserves_completed_candidate_before_gate(self):
        def check():
            if Runtime.instances and "quality" in Runtime.instances[0].events:
                raise BudgetStop("quality evaluation consumed budget")
        report = self.run_costs(check)
        self.assertFalse(report["eligible"])
        self.assertEqual(len(report["units"]), 3)
        self.assertTrue(all(r["generation_completed"] for r in report["units"]))
        self.assertTrue(all(Path(r["retention"]["files"][0]["path"]).exists() for r in report["units"]))
        self.assertNotIn("quality_passed", report["units"][0])
        self.assertEqual(report["active_unit"]["unit_id"], "round0-keep")

    def test_budget_between_rounds_does_not_reclassify_completed_units(self):
        before = {}
        def check():
            path = self.output / "state_cost.json"
            if not path.exists():
                return
            saved = json.loads(path.read_text())
            if len(saved["units"]) == 3 and all(r.get("complete") for r in saved["units"]):
                before["units"] = saved["units"]
                evidence = [*self.output.glob("units/*.json"), *self.output.glob("partial_units/*.json"),
                            *self.output.rglob("*.npy")]
                before["files"] = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
                raise BudgetStop("round0 complete; no budget to start round1")
        report = self.run_costs(check)
        self.assertEqual(report["status"], "stopped")
        self.assertTrue(report["budget_stopped"])
        self.assertFalse(report["eligible"] or report["completed_cost_matrix"])
        self.assertEqual(report["units"], before["units"])
        self.assertTrue(all(r["status"] == "passed" and r["complete"] for r in report["units"]))
        self.assertEqual(before["files"], {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before["files"]})
        self.assertEqual(report["interruption"]["stage"], "between_units")
        self.assertEqual(report["interruption"]["unit_id"], "round1-flush")
        self.assertFalse(report["interruption"]["started"])
        self.assertFalse((self.output / "partial_units/state_cost-round1-flush.json").exists())
        self.assertEqual(Runtime.instances[0].closed, 1)

    def test_prerequisite_supervisor_and_shape_checked_before_runtime(self):
        for mutation in ({"selected_frames":49}, {"eligible":False}, {"supervisor_failure":"isolation"},
                         {"cache_quality_qualified":False}, {"cache_performance_qualified":False}):
            with self.subTest(mutation=mutation):
                path = self.output / "state_cost.json"
                if path.exists(): path.unlink()
                changed = dict(self.qualified, **mutation)
                d2._write(self.output / "supervisor-qualify.json", changed)
                d2._write(self.output / "qualify.json", self.qualified)
                self.assertEqual(self.run_costs()["status"], "blocked")
                self.assertEqual(Runtime.instances, [])

    def test_source_protocol_identity_and_existing_output_are_immutable(self):
        report = self.run_costs()
        paths = list(self.output.rglob("*.npy"))
        hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.assertEqual(self.run_costs()["cost_identity"], report["cost_identity"])
        self.assertEqual(len(Runtime.instances), 1)
        self.assertEqual(hashes, {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        altered = dict(costs.PROTOCOL, seed=102)
        with patch.object(costs.d2, "_read", return_value=altered):
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                self.run_costs()


if __name__ == "__main__":
    unittest.main()
