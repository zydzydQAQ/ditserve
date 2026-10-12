"""Synthetic phase/quality/capacity contracts. No CUDA or model dependencies."""

import copy
import json
import subprocess
import sys
import unittest

from ditserve.experiments.previs2_policy import (FULL, MIB, build_recovery_table, capacity_feasibility,
    choose_action, enumerate_recovery_actions, select_fixed_state, select_initial_policy)


SHAPE = dict(height=256, width=256, frames=17, steps=32)
PROMPTS = ["a", "b", "c", "d"]
PREFIX = dict(cache=True, threshold=.08, max_continuous=1)


def record(prompt, seed, config=None, step_ms=10, prepare_ms=100, loss=.02, **extra):
    config = dict(PREFIX if config is None else config)
    value = dict(type="quality" if config["cache"] else "quality_reference", status="completed", split="dev",
        device="cpu-fixture", shape=dict(SHAPE), prompt_id=prompt, seed=seed, case_id="%s-%s" % (prompt, seed),
        cache_config=config, quality_scope="whole_request_fixed_policy", elapsed_ms=999999,
        quality=dict(status="ok", finite=True, range_valid=True, lpips_mean=loss if config["cache"] else 0.0),
        events=[dict(kind="prepare", ms=prepare_ms)] +
               [dict(kind="step", step_index=i, ms=step_ms) for i in range(32)] + [dict(kind="decode", ms=20)])
    value.update(extra)
    return value


def grid(config=None, **kwargs):
    return [record(p, s, config, **kwargs) for p in PROMPTS for s in (101, 102)]


def resume_grid(mode="hot", suffix="continue", step=16, **kwargs):
    rows = [record(p, 101, PREFIX if suffix == "continue" else FULL, type="quality",
        prefix_config=dict(PREFIX), pause_step=step, mode=mode, suffix=suffix,
        quality_scope="resume_policy", **kwargs) for p in PROMPTS]
    for r in rows:
        r["quality"]["lpips_mean"] = kwargs.get("loss", .02)
        switches = [dict(kind="snapshot", mode=mode, ms=2), dict(kind="restore", mode=mode, ms=3)]
        r["events"][step+1:step+1] = switches + [dict(kind="cache_policy_switch", ms=0)]
        r["switches"] = copy.deepcopy(switches)
    return rows


def table(rows):
    return build_recovery_table(rows, device="cpu-fixture", shape=SHAPE, prefix_config=PREFIX,
                                dev_prompt_ids=PROMPTS, first_seed=101)


def request(**updates):
    value = dict(device="cpu-fixture", shape=dict(SHAPE), seed=102, step_index=16,
        initial_cache_config=dict(PREFIX), selected_cache_config=dict(PREFIX), policy_change_count=0,
        lossy_choices_used=0, recovery_action_used=False, predicted_trajectory_lpips=.03, remaining_ms=180,
        state_cost_ms={m: dict(snapshot_ms=1, restore_ms=1) for m in ("keep", "hot", "cold")})
    value.update(updates)
    return value


def capacity(**overrides):
    values = dict(total_bytes=10000*MIB, driver_free_bytes=1000*MIB, allocated_bytes=7000*MIB,
        reserved_bytes=7500*MIB, victim_storage_bytes=1000*MIB, next_peak_bytes=900*MIB)
    values.update(overrides)
    return capacity_feasibility(**values)


class InitialPolicyTests(unittest.TestCase):
    def select(self, rows, **kwargs):
        result = select_initial_policy(rows, device="cpu-fixture", shape=SHAPE, dev_prompt_ids=PROMPTS,
                                       seeds=(101, 102), **kwargs)
        json.dumps(result, allow_nan=False)
        return result

    def test_import_does_not_load_torch_numpy_or_read_gpu(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from ditserve.experiments import previs2_policy; "
                                 "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_phase_choice_uses_shared_prepare_not_noisy_e2e(self):
        rows = grid(FULL, prepare_ms=100) + grid(step_ms=5, prepare_ms=100000)
        result = self.select(rows)
        self.assertEqual(result["config"], PREFIX)
        self.assertEqual(result["phase_cost_ms"]["prepare"], 100)
        self.assertEqual(result["phase_cost_ms"]["total"], 280)
        self.assertEqual(result["remaining_by_completed_step_ms"]["16"], 100)
        self.assertEqual(result["predicted_trajectory_lpips"], .02)

    def test_fast_but_bad_quality_does_not_pass_epsilon(self):
        result = self.select(grid(FULL) + grid(step_ms=1, loss=.051))
        self.assertEqual(result["config"], FULL)
        self.assertIn("development_quality_exceeds_epsilon", [r["reason"] for r in result["excluded"]])

    def test_whole_trajectory_max_and_exact_threshold_boundary(self):
        rows = grid(step_ms=5); rows[-1]["quality"]["lpips_mean"] = .05
        result = self.select(grid(FULL) + rows)
        self.assertEqual(result["predicted_trajectory_lpips"], .05)

    def test_missing_duplicate_unknown_or_incomplete_events_reject_configuration(self):
        for change in ("missing", "duplicate", "unknown", "events", "m3"):
            rows = grid(step_ms=5)
            if change == "missing": rows.pop()
            if change == "duplicate": rows.append(copy.deepcopy(rows[0]))
            if change == "unknown": rows[0]["quality"]["finite"] = False
            if change == "events": rows[0]["events"].pop(5)
            if change == "m3":
                for r in rows: r["cache_config"]["max_continuous"] = 3
            with self.subTest(change=change): self.assertEqual(self.select(grid(FULL)+rows)["config"], FULL)

    def test_heldout_wrong_device_and_resume_do_not_enter_admission(self):
        rows = grid(FULL) + grid(step_ms=5, split="test") + grid(step_ms=1, device="other") + resume_grid()
        self.assertEqual(self.select(rows)["config"], FULL)

    def test_without_reference_cost_stays_unknown(self):
        result = self.select(grid(step_ms=1))
        self.assertEqual(result["config"], FULL)
        self.assertEqual(result["status"], "fallback_unknown_cost")
        self.assertIsNone(result["phase_cost_ms"])

    def test_model_identity_and_shape_cannot_mix(self):
        rows = grid(FULL, model_config={"revision": "good"}) + grid(step_ms=1, model_config={"revision": "other"})
        self.assertEqual(self.select(rows, model_config={"revision": "good"})["config"], FULL)


class CapacityTests(unittest.TestCase):
    def test_margin_and_keep_headroom_include_existing_allocator_pool(self):
        value = capacity()
        self.assertEqual(value["margin_bytes"], 512*MIB)
        self.assertTrue(value["modes"]["keep"]["feasible"])
        self.assertEqual(value["modes"]["keep"]["allocatable_after_margin_bytes"], 988*MIB)

    def test_release_can_enable_next_increment_but_never_claim_driver_return(self):
        value = capacity(next_peak_bytes=1500*MIB)
        self.assertFalse(value["modes"]["keep"]["feasible"])
        for mode in ("hot", "cold"):
            self.assertTrue(value["modes"][mode]["feasible"])
            self.assertIsNone(value["modes"][mode]["expected_driver_free_after_release_bytes"])
            self.assertTrue(value["modes"][mode]["requires_post_release_check"])

    def test_five_percent_larger_than_512_mib(self):
        self.assertEqual(capacity(total_bytes=20000*MIB)["margin_bytes"], 1000*MIB)

    def test_unknown_or_inconsistent_memory_rejects_all_modes(self):
        for extra in (dict(driver_free_bytes=None),dict(next_peak_bytes=float("nan")),
                      dict(victim_storage_bytes=8000*MIB),dict(reserved_bytes=6000*MIB),dict(total_bytes=True)):
            with self.subTest(extra=extra):
                result = capacity(**extra)
                self.assertTrue(all(not r["feasible"] for r in result["modes"].values()))
                json.dumps(result, allow_nan=False)

    def test_explicit_artificial_state_budget_is_additional_gate(self):
        result = capacity(logical_budget_bytes=1800*MIB,logical_live_bytes=1500*MIB,next_state_bytes=500*MIB)
        self.assertTrue(result["diagnostics"]["artificial_logical_pressure"])
        self.assertFalse(result["modes"]["keep"]["feasible"])
        self.assertTrue(result["modes"]["hot"]["feasible"])
        unknown = capacity(logical_budget_bytes=1800*MIB)
        self.assertTrue(all(not m["feasible"] for m in unknown["modes"].values()))


class RecoveryTests(unittest.TestCase):
    def test_build_complete_whole_trajectory_with_switch_events(self):
        value = table(resume_grid(mode="cold", suffix="full_compute", step_ms=4))
        self.assertEqual(value["status"], "ready")
        a=value["actions"][0]
        self.assertEqual(a["remaining_ms"], 16*4+20)
        self.assertEqual(a["estimated_total_ms"], 16*4+20+2+3)
        self.assertEqual(a["predicted_trajectory_lpips"], .02)
        self.assertEqual(a["config"], FULL)
        json.dumps(value,allow_nan=False)

    def test_seed_102_not_hardcoded_out_and_progress_eight_sixteen_twentyfour(self):
        value=table(sum((resume_grid(mode="cold",step=s) for s in (8,16,24)),[]))
        for s in (8,16,24):
            actions=enumerate_recovery_actions(request(step_index=s),value,capacity())["actions"]
            self.assertEqual(sum(a["mode"]=="cold" for a in actions),1)

    def test_missing_duplicate_and_unknown_quality_reject_calibration(self):
        for change in ("missing","duplicate","unknown","over_budget","wrong_prefix"):
            rows=resume_grid(mode="cold")
            if change=="missing":rows.pop()
            if change=="duplicate":rows.append(copy.deepcopy(rows[0]))
            if change=="unknown":rows[0]["quality"]["status"]="unknown"
            if change=="over_budget":rows[0]["quality"]["lpips_mean"] = .051
            if change=="wrong_prefix":
                for r in rows:r["prefix_config"]["threshold"] = .12
            with self.subTest(change=change):self.assertEqual(table(rows)["actions"],[])

    def test_observed_pause_and_actual_step_configuration_must_match_labels(self):
        for change in ("step", "mode", "threshold"):
            rows = resume_grid(mode="cold")
            if change == "step":
                for r in rows: r["pause_step"] = 8
            if change == "mode": rows[0]["events"][17]["mode"] = "hot"
            if change == "threshold": rows[0]["events"][1]["threshold"] = .12
            with self.subTest(change=change):
                self.assertEqual(table(rows)["actions"], [])

    def test_nonfinite_fallback_cost_is_unknown_json_safe(self):
        r = request(state_cost_ms={"keep": dict(snapshot_ms=float("nan"), restore_ms=1),
                                  "hot": dict(snapshot_ms=1, restore_ms=float("inf"))})
        result = enumerate_recovery_actions(r, table([]), capacity())
        json.dumps(result, allow_nan=False)
        self.assertTrue(all(a["estimated_total_ms"] is None for a in result["actions"]))

    def test_unknown_policy_history_or_nonboolean_capacity_cannot_authorize_cold(self):
        r = request(); del r["policy_change_count"]
        value = table(resume_grid("cold"))
        self.assertFalse(any(a["mode"] == "cold" for a in enumerate_recovery_actions(r, value, capacity())["actions"]))
        cap = capacity(); cap["modes"]["cold"]["feasible"] = "yes"
        self.assertFalse(any(a["mode"] == "cold" for a in enumerate_recovery_actions(request(), value, cap)["actions"]))

    def test_uncovered_hot_continuation_preserves_prior_quality_at_any_step(self):
        value=enumerate_recovery_actions(request(step_index=13,predicted_trajectory_lpips=.047),table([]),capacity())
        hot=next(a for a in value["actions"] if a["mode"]=="hot")
        self.assertEqual(hot["predicted_trajectory_lpips"],.047)
        self.assertFalse(hot["consumes_recovery_choice"])
        self.assertFalse(any(a["mode"]=="cold" for a in value["actions"]))

    def test_calibrated_hot_continuation_has_one_cost_and_does_not_reset_prior_loss(self):
        actions=enumerate_recovery_actions(request(predicted_trajectory_lpips=.047),
            table(resume_grid("hot",loss=.02)),capacity())["actions"]
        hot=[a for a in actions if a["mode"]=="hot" and a["suffix"]=="continue"]
        self.assertEqual(len(hot),1)
        self.assertEqual(hot[0]["predicted_trajectory_lpips"],.047)
        self.assertEqual(hot[0]["estimated_total_ms"],185)
        unknown=enumerate_recovery_actions(request(predicted_trajectory_lpips=None),
            table(resume_grid("hot")),capacity())["actions"]
        self.assertIsNone(next(a for a in unknown if a["mode"]=="hot")["predicted_trajectory_lpips"])

    def test_second_changed_choice_is_not_recalibrated_or_quality_reset(self):
        value=table(resume_grid(mode="cold",suffix="full_compute"))
        for extra in (dict(lossy_choices_used=1),dict(recovery_action_used=True),dict(policy_change_count=1),
                      dict(initial_cache_config=dict(FULL))):
            actions=enumerate_recovery_actions(request(**extra),value,capacity())["actions"]
            self.assertFalse(any(a["mode"]=="cold" or a["suffix"]!="continue" for a in actions))
            self.assertTrue(all(a["predicted_trajectory_lpips"]==.03 for a in actions))

    def test_unknown_quality_and_cost_are_not_synthesized(self):
        actions=enumerate_recovery_actions(request(predicted_trajectory_lpips=None,state_cost_ms={}),table([]),capacity())["actions"]
        self.assertTrue(all(a["predicted_trajectory_lpips"] is None and a["estimated_total_ms"] is None for a in actions))
        chosen=choose_action("edf",actions)
        self.assertEqual(chosen["status"],"fallback_unknown_cost")
        self.assertEqual(chosen["action"]["mode"],"keep")

    def test_unknown_capacity_allows_no_actions(self):
        self.assertEqual(enumerate_recovery_actions(request(),table(resume_grid()),capacity(driver_free_bytes=None))["actions"],[])

    def test_fixed_mode_comparison_uses_common_progress(self):
        rows=resume_grid("hot",step_ms=5)+resume_grid("cold",step_ms=8)+resume_grid("cold",step=8,step_ms=1)
        selection=select_fixed_state(table(rows))
        self.assertEqual(selection["mode"],"hot")
        self.assertEqual(selection["compared_steps"],[16])

    def test_fixed_keep_priority_joint_cost_and_edf_unchanged_suffix(self):
        actions=enumerate_recovery_actions(request(),table(resume_grid("cold",step_ms=1)),capacity())["actions"]
        self.assertEqual(choose_action("fixed_state_qc",actions,fixed_state="cold")["action"]["mode"],"keep")
        self.assertEqual(choose_action("joint_state_qc",actions)["action"]["mode"],"cold")
        self.assertEqual(choose_action("edf",actions)["action"]["suffix"],"continue")

    def test_fixed_cold_uncovered_falls_back_hot_explicitly(self):
        cap=capacity(next_peak_bytes=1500*MIB)
        actions=enumerate_recovery_actions(request(step_index=13),table([]),cap)["actions"]
        selected=choose_action("fixed_state_qc",actions,fixed_state="cold")
        self.assertEqual(selected["action"]["mode"],"hot")
        self.assertIn("uncovered",selected["fallback_reason"])

    def test_hot_unknown_cost_is_preserved_when_release_is_necessary(self):
        actions=enumerate_recovery_actions(request(state_cost_ms={}),table([]),capacity(next_peak_bytes=1500*MIB))["actions"]
        choice=choose_action("joint_state_qc",actions)
        self.assertEqual(choice["status"],"fallback_unknown_cost")
        self.assertIsNone(choice["action"]["estimated_total_ms"])


if __name__ == "__main__":
    unittest.main()
