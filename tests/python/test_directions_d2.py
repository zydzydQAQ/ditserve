"""CPU-only admission/evidence tests for the finite direction-2 experiment."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.directions import direction2 as d2


class BudgetStop(RuntimeError):
    pass


class FakeOOM(RuntimeError):
    pass


def config():
    return dict(models=dict(wan=dict(id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        revision="0fad780a534b6463e45facd96134c9f345acfa5b", path="/offline/model")),
        device="cuda:0", direction2=dict(quality_epsilon=.05, recipe="paired_008"))


def metric(frames, value=.01):
    return dict(status="ok", finite=True, range_valid=True, frame_count=frames,
                shape=[1, frames, 480, 832, 3], lpips_status="ok",
                lpips_mean=value, lpips_p95=value, lpips_per_frame=[value] * frames)


class FakeRuntime:
    instances = []
    bad_frames = set()
    bad_holdout = False
    capacity_at = None
    prefix_oom = False
    cache_ms = 70.
    calls = []

    def __init__(self, configuration, settings, check):
        self.check, self.closed = check, 0
        self.instances.append(self)

    def generate(self, item, frames, policy, diagnostics=False):
        self.check()
        self.calls.append((item["prompt_id"], item["seed"], frames, policy, diagnostics))
        error = frames in self.bad_frames or self.bad_holdout and item["split"] == "holdout"
        output = np.full((1, frames, 1, 1, 3), .2 if error and policy != "off" else .01,
                         dtype=np.float32)
        return dict(output=output, item=item, frames=frames, policy=policy,
                    performance_eligible=not diagnostics,
                    generation_ms=100. if policy == "off" else self.cache_ms,
                    output_sha256="%s-%d-%s" % (item["prompt_id"], item["seed"], policy),
                    path_statistics={}, boundaries=[dict(step=15, inventory={}, refresh_eligible=False),
                                                    dict(step=16, inventory={}, refresh_eligible=True)])

    def quality(self, reference, candidate):
        return metric(reference.shape[1], float(candidate.max()))

    def pressure_trial(self, items, frames, count, urgent_item):
        if self.prefix_oom:
            return dict(count=count, status="failed", pressure_found=False, oom_stage="background_prefix")
        pressure = self.capacity_at is not None and count >= self.capacity_at
        return dict(count=count, status="failed" if pressure else "passed", pressure_found=pressure,
                    urgent_output_sha256="%s-%d-off" % (urgent_item["prompt_id"], urgent_item["seed"]))

    def action_trial(self, items, frames, count, urgent_item, action, references, *, retain_failure=None):
        return dict(action=action, count=count, status="passed", quality_passed=True,
                    all_requests_completed=True, all_requests_ms=100., urgent_ms=10.)

    @staticmethod
    def is_oom(error):
        return isinstance(error, FakeOOM)

    def close(self):
        self.closed += 1


class Direction2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "direction2"
        self.cfg = config()
        FakeRuntime.instances, FakeRuntime.calls, FakeRuntime.bad_frames = [], [], set()
        FakeRuntime.bad_holdout = FakeRuntime.prefix_oom = False
        FakeRuntime.capacity_at, FakeRuntime.cache_ms = None, 70.
        self.factory = patch.object(d2, "_runtime_factory", FakeRuntime)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_phase(self, phase, check=lambda: None):
        return d2.run_phase(self.cfg, self.output, check, phase)

    def qualify(self):
        self.assertTrue(self.run_phase("screen")["eligible"])
        return self.run_phase("qualify")

    def test_current_catalog_and_historical_observations_never_qualify_new_shape(self):
        report = self.run_phase("screen")
        self.assertEqual(report["selected_frames"], 81)
        self.assertEqual(len(report["units"]), 6)
        self.assertTrue(report["eligible"])
        self.assertFalse(report["cache_performance_qualified"])
        self.assertFalse(report["cache_quality_qualified"])
        self.assertFalse(report["sp_resize_qualified"])
        self.assertFalse(report["historical_evidence"]["inherited_quality_qualification"])
        self.assertEqual({call[2] for call in FakeRuntime.calls}, {49, 81})

    def test_bad_large_shape_does_not_inherit_small_shape_quality(self):
        FakeRuntime.bad_frames = {81}
        report = self.run_phase("screen")
        self.assertEqual(report["selected_frames"], 49)
        self.assertEqual(report["shapes"][1]["observed"], 1)
        self.assertTrue(report["units"][-1]["retention"]["failed_quality"])
        self.assertTrue(Path(report["units"][-1]["retention"]["files"][0]["path"]).exists())

    def test_low_gain_or_unknown_quality_cannot_enter_qualification(self):
        FakeRuntime.cache_ms = 97.
        result = self.run_phase("screen")
        self.assertEqual(result["status"], "stopped")
        self.assertFalse(result["eligible"])
        before = len(FakeRuntime.instances)
        self.assertEqual(self.run_phase("qualify")["status"], "blocked")
        self.assertEqual(before, len(FakeRuntime.instances))

    def test_full_counterbalanced_development_and_heldout_are_required(self):
        report = self.qualify()
        self.assertEqual(len(report["units"]), 20)
        dev = [r for r in report["units"] if r["item"]["split"] == "dev"]
        heldout = [r for r in report["units"] if r["item"]["split"] == "holdout"]
        self.assertEqual(len(dev), 16)
        self.assertEqual(len(heldout), 4)
        self.assertEqual({tuple(r["order"]) for r in dev}, {("off", "paired_008"), ("paired_008", "off")})
        self.assertTrue(report["cache_quality_qualified"])
        self.assertTrue(report["cache_performance_qualified"])

    def test_heldout_failure_stops_without_reselection(self):
        self.run_phase("screen")
        FakeRuntime.bad_holdout = True
        report = self.run_phase("qualify")
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["selected_frames"], 81)
        self.assertFalse(report["cache_quality_qualified"])
        self.assertFalse(report["eligible"])

    def test_budget_interrupt_preserves_partial_pair_and_never_qualifies(self):
        calls = [0]
        def check():
            calls[0] += 1
            if calls[0] == 4:
                raise BudgetStop("test deadline")
        report = self.run_phase("screen", check)
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["eligible"])
        self.assertIsNotNone(report["interrupted_unit"])
        partial = list((self.output / "partial_units").glob("*.json"))
        self.assertEqual(len(partial), 1)
        evidence = json.loads(partial[0].read_text())
        self.assertFalse(evidence["complete"])
        self.assertFalse(evidence["qualification"])
        self.assertEqual(evidence["completed_trajectories"], ["off"])
        self.assertTrue(evidence["retention"]["raw_available"])

    def test_existing_phase_is_not_implicitly_rerun(self):
        first = self.run_phase("screen")
        call_count = len(FakeRuntime.calls)
        second = self.run_phase("screen")
        self.assertEqual(first, second)
        self.assertEqual(len(FakeRuntime.calls), call_count)

    def test_changed_config_cannot_inherit_prior_qualification(self):
        self.run_phase("screen")
        self.cfg["models"]["wan"]["path"] = "/different/model"
        result = self.run_phase("qualify")
        self.assertEqual(result["status"], "failed")
        self.assertIn("prerequisite", result["reason"])
        self.assertFalse(result["eligible"])

    def test_pressure_is_bounded_and_no_capacity_difference_blocks_actions(self):
        self.qualify()
        report = self.run_phase("pressure")
        self.assertEqual(report["status"], "stopped")
        counts = [r["count"] for r in report["units"] if "count" in r]
        self.assertEqual(counts, [1, 2, 4, 8, 16])
        self.assertEqual(self.run_phase("actions")["status"], "blocked")

    def test_prefix_oom_does_not_establish_urgent_capacity_opportunity(self):
        self.qualify()
        FakeRuntime.prefix_oom = True
        report = self.run_phase("pressure")
        self.assertFalse(report["eligible"])
        self.assertIn("background-prefix", report["reason"])

    def test_real_pressure_opens_four_baselines_but_not_sp_claim(self):
        self.qualify()
        FakeRuntime.capacity_at = 4
        pressure = self.run_phase("pressure")
        self.assertEqual(pressure["pressure_count"], 4)
        self.assertEqual(pressure["below_pressure_count"], 2)
        report = self.run_phase("actions")
        trials = report["action_trials"]
        self.assertEqual(len(trials), 16)
        self.assertEqual({r["action"] for r in trials}, set(d2.ACTIONS))
        self.assertFalse(report["sp_resize_qualified"])
        self.assertFalse(report["state_mechanism_qualified"])

    def test_path_statistics_use_actual_worker_event_schema(self):
        report = d2._path_statistics([dict(kind="step", completed_steps=9, step_index=8, reused_forwards=2),
                                     dict(kind="step", completed_steps=10, step_index=9, reused_forwards=0)])
        self.assertEqual(report["steps"][0]["step"], 9)
        self.assertEqual(report["reuse_steps"], 1)

    def test_failed_or_partial_frame_metrics_never_pass(self):
        self.assertTrue(d2._quality_known(metric(49), 49))
        for field, value in (("status", "unknown"), ("finite", False),
                             ("frame_count", 48), ("lpips_per_frame", [.01] * 48),
                             ("lpips_mean", float("nan"))):
            candidate = metric(49)
            candidate[field] = value
            self.assertFalse(d2._quality_known(candidate, 49), field)

    def test_large_shape_oom_preserves_completed_reference_and_selects_49(self):
        original = FakeRuntime.generate
        def generate(runtime, item, frames, policy, diagnostics=False):
            if frames == 81 and policy == "paired_008":
                raise FakeOOM("finite capacity failure")
            return original(runtime, item, frames, policy, diagnostics)
        with patch.object(FakeRuntime, "generate", generate):
            report = self.run_phase("screen")
        self.assertEqual(report["selected_frames"], 49)
        self.assertTrue(report["units"][-1]["capacity_failed"])
        self.assertTrue(report["units"][-1]["retention"]["raw_available"])

    def test_direct_action_executor_offloads_and_restores_before_background_suffix(self):
        runtime, transitions = flow_runtime()
        items = d2._inputs("dev")[:2]
        urgent = d2._inputs("holdout")[0]
        references = flow_references(items + [urgent])
        result = runtime.action_trial(items, 49, 2, urgent, "full_cpu", references)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(transitions["urgent_residencies"], ["cpu", "cpu"])
        self.assertEqual(transitions["snapshot_modes"], ["hot", "hot"])
        self.assertEqual(transitions["restore_count"], 2)
        self.assertTrue(result["all_requests_completed"])
        self.assertTrue(result["no_pickle"])

    def test_action_cold_and_reference_warm_prompt_banks_have_identical_arrival_timing(self):
        runtime, _ = flow_runtime()
        items, urgent = d2._inputs("dev")[:1], d2._inputs("holdout")[0]
        clock, warmed = [0.], {urgent["prompt"]}
        def warm(spec):
            if spec["prompt"] not in warmed:
                clock[0] += 2.3  # An observable CPU-encoding setup cost.
                warmed.add(spec["prompt"])
        runtime.bank = SimpleNamespace(warm=warm, entries=warmed)
        original_prepare, original_generate, original_step = runtime.prepare, runtime.generate, runtime.step
        def prepare(item, frames, policy):
            runtime.bank.warm(runtime.spec(item, frames, policy))
            return original_prepare(item, frames, policy)
        def generate(item, frames, policy):
            runtime.bank.warm(runtime.spec(item, frames, policy))
            clock[0] += .25
            return original_generate(item, frames, policy)
        def step(state, policy):
            clock[0] += .01
            return original_step(state, policy)
        runtime.prepare, runtime.generate, runtime.step = prepare, generate, step
        runtime.close = warmed.clear
        with patch.object(d2.time, "perf_counter", side_effect=lambda: clock[0]):
            first = runtime.action_trial(items, 49, 1, urgent, "keep", flow_references(items + [urgent]))
            self.assertFalse(warmed)
            second = runtime.action_trial(items, 49, 1, urgent, "keep", flow_references(items + [urgent]))
        self.assertEqual(first["urgent_encoding_setup_ms"], 0.)
        self.assertAlmostEqual(second["urgent_encoding_setup_ms"], 2300.)
        for name in ("urgent_ms", "all_requests_ms"):
            self.assertAlmostEqual(first[name], second[name])
        self.assertAlmostEqual(first["urgent_ms"], 250.)
        self.assertAlmostEqual(first["all_requests_ms"], 410.)
        self.assertTrue(first["all_requests_completed"] and second["all_requests_completed"])

    def test_action_oom_stage_and_completed_outputs_are_preserved(self):
        items, urgent = d2._inputs("dev")[:1], d2._inputs("holdout")[0]
        for failing in ("setup", "prefix", "action", "urgent", "restore", "suffix", "decode", "quality", "wait_decode"):
            with self.subTest(failing=failing):
                runtime, _ = flow_runtime()
                def fail(*args, **kwargs):
                    raise FakeOOM("injected " + failing)
                action = "keep"
                if failing == "setup":
                    runtime.bank.warm = fail
                elif failing == "prefix":
                    runtime.prepare = fail
                elif failing == "action":
                    action, runtime.state_adapter.snapshot = "full_cpu", fail
                elif failing == "urgent":
                    runtime.generate = fail
                elif failing == "restore":
                    action, runtime.state_adapter.restore = "full_cpu", fail
                elif failing == "suffix":
                    original_step = runtime.step
                    def step(state, policy):
                        if state.step_index >= 16:
                            fail()
                        return original_step(state, policy)
                    runtime.step = step
                elif failing in ("decode", "wait_decode"):
                    runtime.worker.finish = fail
                    if failing == "wait_decode":
                        action = "wait_refresh"
                        runtime.state_adapter.certificate = lambda state: dict(eligible=False)
                elif failing == "quality":
                    runtime.quality = fail
                retained = []
                def retain(runs, details):
                    retained.append((runs, details))
                    return dict(raw_available=bool(runs))
                result = runtime.action_trial(items, 49, 1, urgent, action, flow_references(items + [urgent]),
                                              retain_failure=retain)
                expected = {"restore": "resume", "suffix": "resume", "decode": "finish",
                            "quality": "finish", "wait_decode": "finish"}.get(failing, failing)
                self.assertTrue(result["oom"])
                self.assertEqual(result["oom_stage"], expected)
                self.assertEqual(retained[0][1]["stage"], expected)
                self.assertEqual(result["background_completed"], int(failing == "quality"))
                urgent_completed = failing in ("restore", "suffix", "decode", "quality")
                self.assertEqual(result["urgent_completed"], urgent_completed)
                if urgent_completed:
                    self.assertTrue(any(key.startswith("urgent-candidate-") for key in retained[0][0]))
                if failing == "quality":
                    self.assertTrue(result["all_requests_completed"])
                    self.assertTrue(any(key.startswith("candidate-") for key in retained[0][0]))

    def test_admission_observation_requires_both_keep_ooms_to_be_urgent(self):
        qualified = self.qualify()
        FakeRuntime.capacity_at = 1
        pressure = self.run_phase("pressure")
        for first_stage in ("setup", "prefix", "action", "resume", "finish", None, "urgent"):
            with self.subTest(first_stage=first_stage):
                seen = []
                def trial(runtime, items, frames, count, urgent_item, action, references, *, retain_failure=None):
                    if action == "keep":
                        stage = first_stage if not seen else "urgent"
                        seen.append(stage)
                        return dict(action=action, count=count, status="failed", quality_passed=False,
                                    all_requests_completed=False, oom=True, oom_stage=stage)
                    return dict(action=action, count=count, status="passed", quality_passed=True,
                                all_requests_completed=True, all_requests_ms=100., urgent_ms=10.)
                experiment = d2._Experiment(self.cfg, self.output / str(first_stage), lambda: None, "actions")
                with patch.object(experiment, "previous", side_effect=lambda phase:
                        pressure if phase == "pressure" else qualified), \
                     patch.object(FakeRuntime, "action_trial", trial):
                    experiment.actions()
                expected = set(d2.ACTIONS) - {"keep"} if first_stage == "urgent" else set()
                self.assertEqual(set(experiment.report["observed_admission_actions"]), expected)
                self.assertEqual(len(seen), 2)
                self.assertEqual(len(experiment.report["action_trials"]), 8)

    def test_wait_refresh_advances_actual_requests_before_reclaiming(self):
        runtime, transitions = flow_runtime()
        items, urgent = d2._inputs("dev")[:1], d2._inputs("holdout")[0]
        result = runtime.action_trial(items, 49, 1, urgent, "wait_refresh", flow_references(items + [urgent]))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(transitions["snapshot_modes"], ["exact_refresh"])
        self.assertEqual(transitions["snapshot_steps"], [17])
        self.assertEqual([r["waited_steps"] for r in result["actions"] if "waited_steps" in r], [1])

    def test_urgent_oom_preserves_actual_paused_footprints_and_memory(self):
        runtime, _ = flow_runtime()
        observed = []
        def inventory(state):
            self.assertEqual(state.step_index, 16)
            index = len(observed) + 1
            result = dict(request_id="real-%d" % index, necessary_storage_bytes=100 * index,
                          feature_only_bytes=200 * index, cuda_bytes=280 * index,
                          feature_only_cuda_bytes=200 * index)
            observed.append(result)
            return result
        runtime.state_adapter.inventory = inventory
        runtime.worker._memory = lambda: dict(allocated_bytes=3000, driver_free_bytes=400)
        def urgent_oom(*args):
            raise FakeOOM("urgent allocation failed")
        runtime.generate = urgent_oom
        result = runtime.pressure_trial(d2._inputs("dev")[:2], 49, 2, d2._inputs("holdout")[0])
        self.assertTrue(result["pressure_found"])
        self.assertEqual(result["completed_prefixes"], 2)
        footprint = result["paused_state_bytes"]
        self.assertEqual([r["inventory"] for r in footprint["requests"]], observed)
        self.assertEqual(footprint["necessary_storage_bytes"], 300)
        self.assertEqual(footprint["necessary_cuda_bytes"], 240)
        self.assertEqual(footprint["cache_feature_cuda_bytes"], 600)
        self.assertEqual(result["memory_before_urgent"]["allocated_bytes"], 3000)
        self.assertEqual(result["memory_at_failure"]["driver_free_bytes"], 400)

    def test_failed_action_retains_candidate_and_comparison_references(self):
        experiment = d2._Experiment(self.cfg, self.output, lambda: None, "actions")
        items, urgent = d2._inputs("dev")[:1], d2._inputs("holdout")[0]
        for action, pixel, quality in (("keep", .02, .01), ("flush", .01, .2)):
            with self.subTest(action=action):
                runtime, _ = flow_runtime()
                runtime.worker.finish = lambda state: np.full((1, 49, 1, 1, 3), pixel, dtype=np.float32)
                runtime.quality = lambda reference, output: metric(49, quality)
                result = runtime.action_trial(items, 49, 1, urgent, action, flow_references(items + [urgent]),
                    retain_failure=lambda runs, details: experiment.retain(action, runs, details, failure=True))
                self.assertFalse(result["quality_passed"])
                retention = result["retention"]
                self.assertTrue(retention["raw_available"])
                files = {row["policy"]: Path(row["path"]) for row in retention["files"]}
                candidate = next(path for name, path in files.items() if name.startswith("candidate-"))
                self.assertTrue(np.allclose(np.load(candidate, allow_pickle=False), pixel))
                self.assertTrue(any(name.startswith("reference-off-") for name in files))
                self.assertTrue(any(name.startswith("reference-paired_008-") for name in files))

    def test_action_reference_budget_interrupt_keeps_completed_off_output(self):
        self.qualify()
        FakeRuntime.capacity_at = 1
        self.run_phase("pressure")
        original = FakeRuntime.generate
        def generate(runtime, item, frames, policy, diagnostics=False):
            if policy == "paired_008":
                raise BudgetStop("action reference interrupted after off")
            return original(runtime, item, frames, policy, diagnostics)
        with patch.object(FakeRuntime, "generate", generate):
            report = self.run_phase("actions")
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["eligible"])
        partial = self.output / "partial_units" / "actions-reference-kettle-101.json"
        evidence = json.loads(partial.read_text())
        self.assertEqual(evidence["completed_trajectories"], ["off"])
        self.assertFalse(evidence["qualification"])
        retained = evidence["retention"]["files"][0]
        self.assertEqual(retained["policy"], "off")
        self.assertTrue(Path(retained["path"]).is_file())

    def test_interrupted_action_preserves_completed_urgent_and_background(self):
        runtime, _ = flow_runtime()
        items, urgent = d2._inputs("dev")[:2], d2._inputs("holdout")[0]
        original_finish = runtime.worker.finish
        completed = []
        def finish(state):
            if completed:
                raise BudgetStop("second background interrupted")
            completed.append(state)
            return original_finish(state)
        runtime.worker.finish = finish
        saved = []
        def retain(runs, details):
            saved.append((runs, details))
            return dict(raw_available=True)
        with self.assertRaises(BudgetStop):
            runtime.action_trial(items, 49, 2, urgent, "keep", flow_references(items + [urgent]),
                                 retain_failure=retain)
        self.assertEqual(len(saved), 1)
        runs, details = saved[0]
        self.assertEqual(details["background_completed"], 1)
        self.assertTrue(details["urgent_completed"])
        self.assertTrue(any(name.startswith("urgent-candidate-") for name in runs))
        self.assertTrue(any(name.startswith("candidate-") for name in runs))


def flow_references(items):
    return {d2._case(item): {p: np.full((1, 49, 1, 1, 3), .01, dtype=np.float32)
                            for p in ("off", "paired_008")} for item in items}


def flow_runtime():
    """Exercise the real action orchestration with fake tensor/worker services."""
    transitions = dict(snapshot_modes=[], snapshot_steps=[], restore_count=0, urgent_residencies=[])
    states = []
    runtime = d2._Runtime.__new__(d2._Runtime)
    runtime.check_budget = lambda: None
    runtime.spec = lambda item, frames, policy: dict(prompt=item["prompt"], frames=frames, policy=policy)
    runtime.bank = SimpleNamespace(warm=lambda spec: None, entries={})

    def prepare(item, frames, policy):
        state = SimpleNamespace(step_index=0, residency="gpu", contexts={}, needs_anchor=False)
        states.append(state)
        return state

    def step(state, policy):
        if state.residency != "gpu":
            raise AssertionError("background continued before restoration")
        state.step_index += 1

    def snapshot(state, mode, **kwargs):
        transitions["snapshot_modes"].append(mode)
        transitions["snapshot_steps"].append(state.step_index)
        if mode == "hot":
            state.residency = "cpu"
        return dict(mode=mode)

    def restore(state):
        transitions["restore_count"] += 1
        state.residency = "gpu"
        return dict(mode="restore")

    def generate(item, frames, policy):
        transitions["urgent_residencies"] = [state.residency for state in states]
        return dict(output=np.full((1, 49, 1, 1, 3), .01, dtype=np.float32))

    runtime.worker = SimpleNamespace(_memory=lambda: {}, _sync=lambda: None,
        finish=lambda state: np.full((1, 49, 1, 1, 3), .01, dtype=np.float32),
        release=lambda state: setattr(state, "residency", "released"))
    runtime.state_adapter = SimpleNamespace(snapshot=snapshot, restore=restore,
        certificate=lambda state: dict(eligible=state.step_index >= 17))
    runtime.prepare, runtime.step, runtime.generate = prepare, step, generate
    runtime.quality = lambda reference, output: metric(49)
    runtime.close = lambda: None
    runtime.is_oom = lambda error: isinstance(error, FakeOOM)
    return runtime, transitions


if __name__ == "__main__":
    unittest.main()
