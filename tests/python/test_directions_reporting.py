"""Report fixtures are CPU-only; no model runtime or GPU is needed."""

import json
from pathlib import Path
import tempfile
import unittest

from ditserve.directions.reporting import (CONTINUE, NO_GAIN, SIMPLE, UNVERIFIED,
    budget_summary, build_report, render_markdown, write_report)
from ditserve.directions.runner import PHASES


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ledger = self.root / "ledger.json"

    def put(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def supervisor(self, direction, phase, **value):
        return self.put("direction%d/supervisor-%s.json" % (direction, phase),
                        dict(status="passed", eligible=True, **value))

    def chain(self, direction, through):
        for phase in PHASES[direction]:
            if phase == through:
                break
            self.supervisor(direction, phase)

    def build(self):
        return build_report(self.root, self.ledger, PHASES, now_monotonic=100.)

    def direction(self, number):
        return self.build()["directions"][number - 1]

    def test_missing_measurements_are_unverified_not_zero_latency(self):
        result = self.build()
        self.assertEqual(result["budget"]["status"], "not_started")
        for direction in result["directions"]:
            self.assertEqual(direction["outcome"]["label"], UNVERIFIED)
            self.assertFalse(direction["outcome"]["final"])
            self.assertTrue(all(p["result"]["status"] == "not_run" for p in direction["phases"]))
        self.assertIsNone(result["directions"][2]["metrics"]["profile"]["measured_fps"])
        self.assertIn("缺测不记为零", render_markdown(result))

    def test_worker_success_is_never_authority(self):
        self.chain(1, "online")
        self.put("direction1/worker-online.json", dict(status="passed", eligible=True,
                 decision="potential_quantum_tradeoff", best_simple_sufficient=False, rows=[]))
        result = self.direction(1)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertTrue(result["phases"][-1]["partial"])

    def test_failed_supervisor_dominates_local_success_and_preserves_samples(self):
        self.chain(1, "online")
        self.put("direction1/worker-online.json", dict(status="passed", eligible=True,
                 decision="potential_quantum_tradeoff", best_simple_sufficient=False,
                 rows=[dict(policy="step", trace_sha256="trace", rows=[{}], elapsed_ms=1)]))
        self.put("direction1/supervisor-online.json", dict(status="incomplete", eligible=False,
                 supervisor_failure=dict(reason="GPU isolation lost")))
        self.put("direction1/worker-config.json", dict(direction1=dict(policies=["step"])))
        result = self.direction(1)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertEqual(result["metrics"]["online"][0]["observed_requests"], 1)
        self.assertTrue(result["phases"][-1]["partial"])

    def test_authoritative_snapshot_wins_over_newer_worker_rows(self):
        self.supervisor(1, "profile", rows=[dict(kind="video_profile", step_ms=[2., 4.], shape={})])
        self.put("direction1/worker-profile.json", dict(rows=[dict(kind="video_profile", step_ms=[100.], shape={})]))
        result = self.direction(1)
        self.assertEqual(result["metrics"]["profile"]["shapes"][0]["step_ms"]["mean"], 3.)
        self.assertFalse(result["phases"][0]["partial"])

    def test_profile_reports_shape_sample_units_and_partial_correctness(self):
        self.supervisor(1, "profile", co_resident=True, rows=[dict(kind="video_profile",
            shape=dict(height=480, width=832, frames=81), status="completed", step_ms=[1., 2., 3., 4., 5., 6.],
            truncated=True, block_cuda=[{}] * 120, extra_state_bytes=[123, 456])])
        self.put("direction1/worker-config.json", dict(direction1=dict(prompts=["a", "b"], seeds=[101, 102])))
        self.put("direction1/direction1-correctness.json", dict(status="running", rows=[
            dict(shape=dict(height=256, width=256, frames=9), passed=True, uncut_split_overhead=.01,
                 step_checks=[{}] * 64, solver_history_checks=[{}] * 32) for _ in range(4)]))
        result = self.direction(1)
        shape = result["metrics"]["profile"]["shapes"][0]
        self.assertEqual(shape["step_ms"], dict(n=6, mean=3.5, p95=5.75, maximum=6.))
        self.assertEqual(shape["block_samples"], 120)
        self.assertEqual(shape["max_extra_state_bytes"], 456)
        correctness = result["metrics"]["correctness"]
        self.assertEqual((correctness["observed"], correctness["expected"], correctness["passed"]), (4, 8, 4))
        self.assertEqual(correctness["solver_checks"], 128)
        self.assertIsNone(correctness["max_uncut_split_overhead"])
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertIn("已完成 4 / 8", render_markdown(self.build()))

    def test_profile_residency_uses_recorded_components_not_full_gpu_assumption(self):
        self.supervisor(1, "profile", co_resident=True, rows=[])
        rendered = render_markdown(self.build())
        self.assertIn("组件位置（execution.residency）：未记录", rendered)
        self.supervisor(1, "profile", co_resident=True, rows=[], execution=dict(residency=dict(
            wan_transformer="cuda:0", wan_vae="cuda:0", sana_all_components="cuda:0",
            wan_text_encoder="cpu_float32_existing_flash_baseline")))
        rendered = render_markdown(self.build())
        self.assertIn("Wan DiT：GPU（cuda:0）", rendered)
        self.assertIn("Wan VAE：GPU（cuda:0）", rendered)
        self.assertIn("Sana 全部组件：GPU（cuda:0）", rendered)
        self.assertIn("Wan 文本编码器：CPU FP32（沿用既有 Flash 基线）", rendered)

    def test_playout_scope_stops_at_independent_headless_consumer(self):
        rendered = render_markdown(self.build())
        self.assertIn("独立实时无界面消费端的出队时间", rendered)
        self.assertIn("不含浏览器、网络或物理显示延迟", rendered)

    def test_preemption_pairing_excludes_unmatched_trial_and_counts_memory(self):
        self.put("direction1/worker-config.json", dict(direction1=dict(policies=["step", "block-1"])))
        rows = [dict(policy=p, round=0, fraction=f, image_wait_ms=wait, max_extra_state_bytes=memory)
                for p, f, wait, memory in (("step", .1, 10., 0), ("block-1", .1, 2., 64), ("step", .5, 100., 0))]
        self.put("direction1/direction1-preemption.json", dict(status="running", rows=rows))
        result = self.direction(1)["metrics"]["preemption"]
        self.assertEqual(result[0]["observed"], 2)
        self.assertEqual(result[0]["paired_n"], 1)
        self.assertEqual(result[0]["image_wait_ms"]["p95"], 10.)
        self.assertEqual(result[1]["paired_wait_delta_ms"]["mean"], -8.)
        self.assertEqual(result[1]["max_extra_state_bytes"], 64)

    def test_online_uses_frozen_trace_pairs_and_keeps_unpaired_denominator(self):
        self.put("direction1/worker-config.json", dict(direction1=dict(policies=["step", "cfg"])))
        rows = [dict(policy=p, trace_sha256=trace, elapsed_ms=100.,
                     rows=[dict(kind="image", status="completed", arrival_ms=0., start_ms=wait,
                                finish_ms=wait + 1., deadline_ms=100.) for _ in range(count)],
                     image_metrics=dict(wait_p95_ms=wait), metrics=dict(goodput_rps=2),
                     video_metrics=dict(on_time_fraction=1))
                for p, trace, count, wait in (("step", "a", 3, 10), ("cfg", "a", 3, 4), ("cfg", "b", 2, 1))]
        self.put("direction1/direction1-online.json", dict(rows=rows,
                 active_trace=dict(rows=[dict(status="pending")], complete=False)))
        result = self.direction(1)["metrics"]
        cfg = result["online"][1]
        self.assertEqual((cfg["observed_replays"], cfg["observed_requests"], cfg["paired_traces"]), (2, 5, 1))
        self.assertEqual(cfg["paired_deltas_vs_step"]["image_wait_p95_ms"]["mean"], -6)
        self.assertIsNone(cfg["gate_metrics"])
        self.assertFalse(result["active_trace"]["complete"])

    def test_interrupted_online_derives_absolute_metrics_from_complete_pairs_only(self):
        policies = ["step", "cfg", "block-1", "block-4", "block-8"]
        self.put("direction1/worker-config.json", dict(direction1=dict(policies=policies, online_count=2,
            online_seeds=[101, 102, 103], online_loads=[.5, .9], patterns=["ordinary", "burst"])))
        first = [dict(kind="image", status="completed", arrival_ms=100., start_ms=120., finish_ms=150., deadline_ms=200.),
                 dict(kind="video", status="completed", arrival_ms=0., start_ms=0., finish_ms=800., deadline_ms=700.)]
        second = [dict(kind="image", status="completed", arrival_ms=0., start_ms=40., finish_ms=80., deadline_ms=70.),
                  dict(kind="video", status="completed", arrival_ms=10., start_ms=20., finish_ms=2500., deadline_ms=2600.)]
        rows = [dict(policy=policy, trace_sha256=trace, rows=requests, elapsed_ms=elapsed,
                     metrics=dict(goodput_rps=99999.), image_metrics=dict(wait_p95_ms=99999.))
                for trace, requests, elapsed in (("first", first, 1000.), ("second", second, 3000.))
                for policy in policies]
        # An unpaired completed replay and an all-policy group containing one
        # interrupted replay must not dilute any paired policy denominator.
        rows.append(dict(policy="step", trace_sha256="unpaired", rows=first, elapsed_ms=1000.))
        for policy in policies:
            requests = [dict(r) for r in first]
            if policy == "block-8":
                requests[1].update(status="interrupted", finish_ms=None)
            rows.append(dict(policy=policy, trace_sha256="interrupted", rows=requests, elapsed_ms=1000.))
        self.put("direction1/supervisor-online.json", dict(status="incomplete", eligible=False, rows=rows,
            reason="budget exhausted", active_trace=dict(policy="cfg", trace_sha256="active", rows=second,
                                                          elapsed_ms=3000., complete=False)))
        result = self.direction(1)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertEqual(result["metrics"]["online_pair_coverage"], dict(expected=12, observed=2,
                         observed_trace_groups=4, excluded_incomplete_groups=2, excluded_complete_replays=5))
        for policy in result["metrics"]["online"]:
            self.assertEqual(policy["paired_traces"], 2)
            self.assertIsNone(policy["gate_metrics"])
            aggregate = policy["descriptive_aggregate"]
            self.assertTrue(aggregate["descriptive_only"])
            self.assertFalse(aggregate["qualified"])
            self.assertEqual(aggregate["elapsed_ms_including_drain"], 4000.)
            self.assertEqual(aggregate["overall"]["arrived"], 4)
            self.assertEqual(aggregate["overall"]["on_time"], 2)
            self.assertEqual(aggregate["overall"]["goodput_rps"], .5)
            self.assertEqual(aggregate["image"]["wait_ms"], dict(n=2, mean=30., p95=39., maximum=40.))
            self.assertEqual(aggregate["image"]["latency_ms"], dict(n=2, mean=65., p95=78.5, maximum=80.))
            self.assertEqual(aggregate["video"]["on_time_fraction"], .5)
        rendered = render_markdown(self.build())
        self.assertIn("完整配对 2 / 预期 12", rendered)
        self.assertIn("不形成最终正负结论", rendered)
        self.assertIn("30.00 / 39.00 / 2", rendered)
        self.assertIn("65.00 / 78.50 / 2", rendered)
        self.assertIn("50.00%", rendered)

    def test_budget_worker_envelope_preserves_57_native_replays_and_interruption(self):
        policies = ["step", "cfg", "block-1", "block-4", "block-8"]
        self.chain(1, "online")
        self.put("direction1/worker-config.json", dict(direction1=dict(policies=policies, online_count=24,
            online_seeds=[101, 102, 103], online_loads=[.5, .9], patterns=["ordinary", "burst"])))
        rows = []
        for trace in range(12):
            for policy_index, policy in enumerate(policies if trace < 11 else policies[:2]):
                requests = []
                for index in range(24):
                    video = index in (0, 12)
                    wait = 20. + policy_index if trace < 11 else 100000.
                    requests.append(dict(request_id="r%03d" % index,
                        kind="video" if video else "image", status="completed", arrived=True,
                        arrival_ms=0., start_ms=0. if video else wait,
                        finish_ms=800. if video else wait + 10., deadline_ms=900. if video else 50.))
                rows.append(dict(policy=policy, trace_sha256="trace-%d" % trace, rows=requests,
                                 elapsed_ms=1000. + trace * 10. if trace < 11 else 200000.))
        active = dict(policy="block-1", trace_sha256="trace-11", complete=False, elapsed_ms=25.55,
            rows=[dict(request_id="r%03d" % index, kind="video" if index in (0, 12) else "image",
                       status="interrupted" if index == 0 else "pending", arrived=index == 0,
                       start_ms=19.97 if index == 0 else None, finish_ms=None) for index in range(24)])
        envelope = dict(status="incomplete", eligible=False, budget_stopped=True,
                        reason="stage allowance reached; stop before starting another unit")
        authority = self.put("direction1/supervisor-online.json", dict(envelope, returncode=2))
        self.put("direction1/worker-online.json", envelope)
        native = self.put("direction1/direction1-online.json", dict(status="incomplete", rows=rows,
            active_trace=active, best_simple_sufficient=True, summaries={p: dict(meets_screen=True) for p in policies}))
        report = self.build()
        result = report["directions"][0]
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertFalse(result["outcome"]["final"])
        phase = result["phases"][-1]
        self.assertEqual(phase["authority"], str(authority))
        self.assertEqual(phase["measurement_source"], str(native))
        self.assertTrue(phase["result"]["budget_stopped"])
        self.assertTrue(phase["partial"])
        metrics = result["metrics"]
        self.assertEqual(metrics["online_pair_coverage"], dict(expected=12, observed=11,
            observed_trace_groups=12, excluded_incomplete_groups=1, excluded_complete_replays=2))
        self.assertEqual(metrics["active_trace"], active)
        self.assertIsNone(metrics["best_simple_sufficient"])
        for index, policy in enumerate(metrics["online"]):
            observed = 12 if index < 2 else 11
            self.assertEqual((policy["observed_replays"], policy["observed_requests"]), (observed, observed * 24))
            self.assertEqual(policy["paired_traces"], 11)
            self.assertIsNone(policy["gate_metrics"])
            aggregate = policy["descriptive_aggregate"]
            self.assertTrue(aggregate["descriptive_only"])
            self.assertFalse(aggregate["qualified"])
            self.assertEqual(aggregate["elapsed_ms_including_drain"], 11550.)
            self.assertEqual(aggregate["overall"]["arrived"], 264)
            self.assertEqual(aggregate["overall"]["on_time"], 264)
            self.assertAlmostEqual(aggregate["overall"]["goodput_rps"], 264000. / 11550.)
            self.assertEqual(aggregate["image"]["wait_ms"], dict(n=242, mean=20. + index,
                p95=20. + index, maximum=20. + index))
            self.assertEqual(aggregate["video"]["sample_count"], 22)
        rendered = render_markdown(report)
        self.assertIn("完整配对 11 / 预期 12", rendered)
        self.assertIn("2 个完整但未配对的 replay", rendered)
        self.assertIn("不形成最终正负结论", rendered)

    def test_explicit_empty_worker_measurements_are_not_replaced_by_native(self):
        worker = self.put("direction1/worker-online.json", dict(status="incomplete", rows=[]))
        self.put("direction1/direction1-online.json", dict(rows=[dict(policy="step", trace_sha256="old")]))
        result = self.direction(1)
        self.assertEqual(result["phases"][-1]["measurement_source"], str(worker))
        self.assertEqual(result["metrics"]["online_pair_coverage"]["observed_trace_groups"], 0)

    def test_online_without_valid_complete_pairs_has_no_zero_latency_or_goodput(self):
        self.put("direction1/worker-config.json", dict(direction1=dict(policies=["step", "cfg"], online_count=2)))
        valid = dict(kind="image", status="completed", arrival_ms=0., start_ms=1., finish_ms=2., deadline_ms=3.)
        self.put("direction1/direction1-online.json", dict(status="running", rows=[
            dict(policy="step", trace_sha256="a", rows=[valid, valid], elapsed_ms=10.),
            dict(policy="cfg", trace_sha256="a", rows=[valid], elapsed_ms=10.)]))
        result = self.direction(1)
        for row in result["metrics"]["online"]:
            self.assertEqual(row["paired_traces"], 0)
            self.assertIsNone(row["descriptive_aggregate"]["overall"]["goodput_rps"])
            self.assertIsNone(row["descriptive_aggregate"]["image"]["wait_ms"]["p95"])
            self.assertIsNone(row["gate_metrics"])

    def test_d1_decision_categories_require_supervisor_eligibility(self):
        self.chain(1, "online")
        for status, eligible, decision, sufficient, expected in (
            ("passed", True, "potential_quantum_tradeoff", False, CONTINUE),
            ("stopped", False, "simple_policy_sufficient", True, SIMPLE),
            ("stopped", False, "stop_no_online_benefit", False, NO_GAIN),
            ("failed", False, "simple_policy_sufficient", True, UNVERIFIED),
            ("passed", False, "potential_quantum_tradeoff", False, UNVERIFIED)):
            with self.subTest(decision=decision, status=status, eligible=eligible):
                self.put("direction1/supervisor-online.json", dict(status=status, eligible=eligible,
                    decision=decision, best_simple_sufficient=sufficient))
                self.assertEqual(self.direction(1)["outcome"]["label"], expected)

    def test_prior_supervisor_failure_prevents_later_success(self):
        self.chain(1, "online")
        self.put("direction1/supervisor-correctness.json", dict(status="failed", eligible=False))
        self.supervisor(1, "online", decision="potential_quantum_tradeoff", best_simple_sufficient=False)
        self.assertEqual(self.direction(1)["outcome"]["label"], UNVERIFIED)

    def test_budget_stop_does_not_become_negative_scientific_finding(self):
        self.chain(1, "online")
        self.put("direction1/admission-online.json", dict(status="stopped", eligible=False,
                 budget_stopped=True, scientific_outcome="not_measured_budget_limited", reason="budget insufficient"))
        self.assertEqual(self.direction(1)["outcome"]["label"], UNVERIFIED)

    def test_d2_quality_failure_and_oom_are_different_outcomes(self):
        self.supervisor(2, "screen")
        for capacity, expected in ((False, NO_GAIN), (True, UNVERIFIED)):
            self.put("direction2/supervisor-qualify.json", dict(status="stopped", eligible=False,
                reason="development quality failed; no retuning", units=[dict(quality_passed=False,
                quality=dict(status="ok", lpips_status="ok", lpips_mean=.09),
                capacity_failed=capacity, runs={"off": dict(generation_ms=3)})]))
            result = self.direction(2)
            self.assertEqual(result["outcome"]["label"], expected)
            self.assertEqual(result["metrics"]["qualify"]["failed_units"], 1)
            self.assertEqual(result["metrics"]["qualify"]["quality_pairs"], 1)
            self.assertEqual(result["metrics"]["qualify"]["capacity_failures"], int(capacity))

    def test_d2_screen_is_not_full_qualification(self):
        self.supervisor(2, "screen", cache_quality_qualified=False, cache_performance_qualified=False,
                        shapes=[dict(frames=81, quality_screen_passed=True, observed=3, required=3, descriptive_gain=.1)])
        result = self.direction(2)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertFalse(result["metrics"]["screen"]["cache_performance_qualified"])

    def test_unknown_lpips_is_unverified_not_negative_quality(self):
        self.supervisor(2, "screen")
        self.put("direction2/supervisor-qualify.json", dict(status="stopped", eligible=False,
            reason="development quality failed; no retuning", units=[dict(quality_passed=False,
                runs={}, quality=dict(status="unknown", lpips_status="unavailable"))]))
        result = self.direction(2)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertEqual(result["metrics"]["qualify"]["unverified_quality_failures"], 1)

    def test_d2_pressure_and_action_scope_preserves_failed_trials(self):
        self.chain(2, "actions")
        self.put("direction2/supervisor-actions.json", dict(status="passed", eligible=False,
            state_mechanism_qualified=False, cache_quality_qualified=False,
            action_trials=[dict(action="keep", count=4, status="failed", oom=True),
                           dict(action="offload", count=4, status="passed", quality_passed=True)],
            units=[dict(action="keep", count=4, status="failed", oom=True),
                   dict(action="offload", count=4, status="passed", quality_passed=True)],
            observed_admission_actions=["offload"], completed_action_matrix=True,
            reason="finite mechanism measurements complete; flush lacks full 8-dev/4-heldout action qualification"))
        result = self.direction(2)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertEqual(result["metrics"]["actions"]["failed_units"], 1)
        self.assertEqual(result["metrics"]["actions"]["action_counts"], [4])
        self.assertEqual(result["metrics"]["actions"]["observed_admission_actions"], ["offload"])

    def test_d2_keep_sufficient_is_scoped_to_measured_pressure_range(self):
        self.chain(2, "pressure")
        self.put("direction2/supervisor-pressure.json", dict(status="stopped", eligible=False,
            reason="keep admits urgent through K=16; no observed capacity opportunity",
            units=[dict(count=k, pressure_found=False, status="passed") for k in (1, 2, 4, 8, 16)]))
        result = self.direction(2)
        self.assertEqual(result["outcome"]["label"], SIMPLE)
        self.assertEqual(len(result["metrics"]["pressure"]["pressure_trials"]), 5)

    def cost_rows(self):
        rows = []
        for round_id in (0, 1):
            for position, action in enumerate(("keep", "full_cpu", "flush")):
                ms = 2. + 2. * round_id
                released = 100. + 100. * round_id if action != "keep" else 0.
                rows.append(dict(unit_id="round%d-%s" % (round_id, action), round=round_id,
                    position=position, action=action, status="passed", complete=True, stage="completed",
                    last_completed_step=32, frames=81, pause_step=16,
                    timings=dict(d2h_ms=ms if action == "full_cpu" else 0.,
                        h2d_ms=ms * 2 if action == "full_cpu" else 0.,
                        flush_ms=ms if action == "flush" else 0., action_ms=ms,
                        next_step_ms=ms * 10, suffix_complete_ms=ms * 100, post_pause_work_ms=ms * 101),
                    transfer=dict(d2h_bytes=1000 if action == "full_cpu" else 0,
                                  h2d_bytes=1000 if action == "full_cpu" else 0),
                    memory=dict(before=dict(allocated_bytes=1000, reserved_bytes=9000),
                                after_action=dict(allocated_bytes=1000-released, reserved_bytes=9000),
                                after_restore=dict(allocated_bytes=1000, reserved_bytes=9000)),
                    quality=dict(status="ok", lpips_status="ok", lpips_mean=.01 + round_id * .01),
                    quality_passed=True, exact_to_keep=action != "flush",
                    comparison=dict(next_step_over_keep_ms=ms), output_sha256="output"))
        return rows

    def test_d2_cost_means_quantiles_and_measured_release_are_descriptive(self):
        self.chain(2, "state_cost")
        self.supervisor(2, "state_cost", units=self.cost_rows(), completed_cost_matrix=True,
            descriptive_only=True, state_mechanism_qualified=False, capacity_qualified=False,
            sp_resize_qualified=False)
        result = self.direction(2)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        cost = result["metrics"]["state_cost"]["cost_measurements"]
        self.assertEqual((cost["observed_trials"], cost["completed_trials"], cost["expected_trials"]), (6, 6, 6))
        self.assertTrue(cost["descriptive_only"])
        self.assertFalse(cost["qualified"])
        full = next(a for a in cost["arms"] if a["action"] == "full_cpu")
        self.assertEqual(full["timings"]["d2h_ms"], dict(n=2, mean=3., p95=3.9, maximum=4.))
        self.assertEqual(full["allocated_release_bytes"], dict(n=2, mean=150., p95=195., maximum=200.))
        self.assertEqual(full["allocated_restore_increase_bytes"], full["allocated_release_bytes"])
        self.assertEqual(full["timings"]["prefix_ms"]["n"], 0)
        self.assertEqual(full["exact_to_keep_passed"], 2)
        flush = next(a for a in cost["arms"] if a["action"] == "flush")
        self.assertEqual((flush["quality_passed"], flush["exact_to_keep_passed"], flush["exact_to_keep_observed"]), (2, 0, 2))
        rendered = render_markdown(self.build())
        self.assertIn("独立状态成本（描述性）", rendered)
        self.assertIn("3.00 / 3.90 / 2", rendered)
        self.assertIn("成本阶段通过不授状态、容量或 SP 科研资格", rendered)

    def test_d2_later_cost_pass_or_failure_cannot_replace_pressure_terminal(self):
        self.chain(2, "pressure")
        self.put("direction2/supervisor-pressure.json", dict(status="stopped", eligible=False,
            reason="keep admits urgent through K=16; no observed capacity opportunity", units=[]))
        # Even consumers that append the late supplementary phase must preserve
        # the scientific gate order rather than use its filesystem timestamp.
        phases = dict(PHASES)
        phases[2] = ("screen", "qualify", "pressure", "actions", "state_cost")
        for status in ("passed", "failed"):
            self.put("direction2/supervisor-state_cost.json", dict(status=status, eligible=status == "passed",
                units=self.cost_rows(), completed_cost_matrix=True, descriptive_only=True,
                returncode=0 if status == "passed" else 1))
            result = build_report(self.root, self.ledger, phases)["directions"][1]
            self.assertEqual(result["outcome"]["label"], SIMPLE)
            self.assertEqual(result["outcome"]["phase"], "pressure")
            self.assertEqual(result["phases"][-1]["result"]["status"], status)
            self.assertEqual(result["metrics"]["state_cost"]["cost_measurements"]["observed_trials"], 6)

    def test_d2_cost_budget_envelope_keeps_partial_operations_and_failure_denominators(self):
        rows = self.cost_rows()[:2]
        rows[0].update(status="failed", quality_passed=False, exact_to_keep=False)
        rows[1].update(status="incomplete", complete=False, stage="restore", quality=None,
                       quality_passed=None, exact_to_keep=None, error="budget stop")
        rows[1]["timings"] = dict(d2h_ms=7., h2d_ms=None, flush_ms=0.)
        rows[1]["memory"] = dict(before=dict(allocated_bytes=1000),
                                  after_action=dict(allocated_bytes=1100), after_restore=None)
        active = dict(unit_id=rows[1]["unit_id"], stage="restore", last_completed_step=16)
        self.put("direction2/state_cost.json", dict(units=rows, active_unit=active, completed_cost_matrix=False))
        envelope = dict(status="incomplete", eligible=False, budget_stopped=True, reason="budget stop")
        self.put("direction2/worker-state_cost.json", envelope)
        self.put("direction2/supervisor-state_cost.json", envelope)
        result = self.direction(2)
        cost = result["metrics"]["state_cost"]["cost_measurements"]
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertEqual((cost["completed_trials"], cost["observed_trials"]), (1, 2))
        self.assertEqual(cost["active_unit"], active)
        keep, full, flush = cost["arms"]
        self.assertEqual((keep["failed"], keep["quality_passed"], keep["quality_observed"]), (1, 0, 1))
        self.assertEqual(full["timings"]["d2h_ms"]["n"], 1)
        self.assertIsNone(full["timings"]["h2d_ms"]["mean"])
        self.assertEqual(full["allocated_release_bytes"]["mean"], -100)
        self.assertEqual(full["allocated_restore_increase_bytes"]["n"], 0)
        self.assertEqual(flush["observed"], 0)
        self.assertIsNone(flush["timings"]["flush_ms"]["mean"])
        rendered = render_markdown(self.build())
        self.assertIn("保留未完成 active_unit", rendered)
        self.assertIn("budget stop", rendered)
        self.assertIn("未完成 / 失败试次", rendered)

    def test_d2_cost_local_pass_cannot_override_failed_pressure_supervisor(self):
        self.chain(2, "pressure")
        self.put("direction2/supervisor-pressure.json", dict(status="failed", eligible=False, returncode=1,
                 reason="isolation failure"))
        self.supervisor(2, "state_cost", units=self.cost_rows(), completed_cost_matrix=True, descriptive_only=True)
        result = self.direction(2)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertFalse(result["outcome"]["final"])

    def test_d2_pressure_preserves_actual_state_and_allocated_without_reserved_fallback(self):
        state = dict(necessary_storage_bytes=41959236, cache_feature_storage_bytes=405700608,
                     necessary_cuda_bytes=41935104, cache_feature_cuda_bytes=405700608)
        memory = dict(allocated_bytes=3830182400, reserved_bytes=16399728640)
        self.put("direction2/pressure.json", dict(units=[dict(count=1, status="passed", pressure_found=False,
                     paused_state_bytes=state, memory_before_urgent=memory),
                     dict(count=2, status="failed", pressure_found=False, oom_stage="background_prefix",
                          memory_before_urgent=dict(reserved_bytes=9999))], active_unit=dict(key="k4")))
        result = self.direction(2)["metrics"]["pressure"]
        self.assertEqual(result["pressure_trials"][0]["paused_state_bytes"], state)
        self.assertEqual(result["pressure_trials"][0]["memory_before_urgent"], memory)
        self.assertIsNone(result["pressure_trials"][1]["paused_state_bytes"])
        rendered = render_markdown(self.build())
        self.assertIn("41959236 / 405700608", rendered)
        self.assertIn("3830182400", rendered)
        self.assertNotIn("9999", rendered)
        self.assertIn("压力探测未完成背景后缀", rendered)

    def test_d3_single_session_capacity_failure_is_unverified_with_fps_evidence(self):
        self.chain(3, "profile")
        reason = "A100_480p_below_16fps_no_fps_relabeling"
        self.put("direction3/supervisor-profile.json", dict(status="stopped", eligible=False,
                 measured_fps=8.2, reason=reason, configured_warmup_blocks=2,
                 effective_warmup_blocks=4, measured_chunk_indices=list(range(4, 10)),
                 samples=[dict(warmup=index < 4) for index in range(10)]))
        result = self.direction(3)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertFalse(result["outcome"]["final"])
        self.assertIn(reason, result["outcome"]["reason"])
        entry = next(e for e in result["phases"] if e["phase"] == "profile")
        self.assertEqual(entry["result"]["status"], "stopped")
        self.assertFalse(entry["result"]["eligible"])
        self.assertEqual(entry["result"]["reason"], reason)
        self.assertFalse(entry["partial"])
        self.assertEqual(result["metrics"]["profile"]["measured_fps"], 8.2)
        self.assertEqual(result["metrics"]["profile"]["required_fps"], 16)
        self.assertEqual(result["metrics"]["profile"]["measured_chunks"], 6)
        rendered = render_markdown(self.build())
        self.assertIn("方向 3：" + UNVERIFIED, rendered)
        self.assertIn("8.20 FPS", rendered)
        self.assertIn(reason, rendered)

    def test_d3_technical_delays_keep_superseded_pending_events(self):
        self.chain(3, "update")
        events = [dict(status="applied", condition_latency_s=.1, conditioned_decode_latency_s=.3,
                       conditioned_playout_latency_s=.8, semantic_latency_s=None),
                  dict(status="superseded"), dict(status="pending")]
        self.supervisor(3, "update", technical_eligible=True, semantic_eligible=False,
            semantic_status="pending_blinded_review_no_semantic_success_claim",
            samples=[dict(policy="replace", updates={"session": events}, playback={"session": dict(
                presentations=[{}] * 12, stall_seconds=.5)})])
        result = self.direction(3)
        arm = result["metrics"]["update"]["arms"][0]
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.assertEqual(arm["events"], 3)
        self.assertEqual(arm["event_statuses"], dict(applied=1, superseded=1, pending=1))
        self.assertEqual(arm["delays_seconds"]["conditioned_playout_latency_s"]["n"], 1)
        self.assertEqual(arm["delays_seconds"]["conditioned_playout_latency_s"]["unobserved"], 2)
        self.assertEqual(arm["delays_seconds"]["conditioned_playout_latency_s"]["mean"], .8)
        self.assertFalse(result["metrics"]["update"]["semantic_eligible"])
        self.assertIn("不能替代语义成功", render_markdown(self.build()))

    def test_d3_two_session_capacity_failure_is_resource_limited(self):
        self.chain(3, "multisession")
        self.put("direction3/supervisor-multisession.json", dict(status="stopped", eligible=False,
                 reason="single_GPU_service_curve_below_two_times_16fps"))
        self.assertEqual(self.direction(3)["outcome"]["label"], UNVERIFIED)

    def test_d3_initial_condition_is_separate_and_session_scheduler_identity_survives(self):
        self.chain(3, "multisession")
        samples = [dict(session_policy=policy, updates={"s": [
            dict(version=0, condition_latency_s=10.), dict(version=1, condition_latency_s=.1),
            dict(version=2, status="superseded")]}, scheduled_events=3, observed_events=3)
            for policy in ("round_robin", "minimum_playback_slack")]
        self.put("direction3/supervisor-multisession.json", dict(status="passed", eligible=False, samples=samples))
        arms = self.direction(3)["metrics"]["multisession"]["arms"]
        self.assertEqual(arms[0]["initial_events"], 1)
        self.assertEqual(arms[0]["update_events"], 2)
        self.assertEqual(arms[0]["events"], 3)
        self.assertEqual(arms[0]["delays_seconds"]["condition_latency_s"]["mean"], .1)
        self.assertEqual(arms[0]["delays_seconds"]["condition_latency_s"]["unobserved"], 1)
        self.assertEqual(arms[0]["initial_delays_seconds"]["condition_latency_s"]["mean"], 10.)
        rendered = render_markdown(self.build())
        self.assertIn("round_robin", rendered)
        self.assertIn("minimum_playback_slack", rendered)

    def test_failed_supervision_masks_local_qualification_flags(self):
        self.put("direction3/worker-update.json", dict(status="passed", eligible=True,
                 technical_eligible=True, semantic_eligible=False, samples=[]))
        self.put("direction3/supervisor-update.json", dict(status="failed", eligible=False))
        result = self.direction(3)
        self.assertIsNone(result["metrics"]["update"]["technical_eligible"])
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        self.put("direction2/qualify.json", dict(status="passed", eligible=True,
                 cache_quality_qualified=True, cache_performance_qualified=True, units=[]))
        result = self.direction(2)
        self.assertIsNone(result["metrics"]["qualify"]["cache_quality_qualified"])

    def test_d3_buffer_peaks_and_stale_fraction_preserve_measured_scope(self):
        self.put("direction3/worker-profile.json", dict(configured_warmup_blocks=2,
                 effective_warmup_blocks=4, measured_chunk_indices=list(range(4, 10))))
        sample = dict(policy="latest_only", buffer_chunks=1, updates={}, playback={
            "s0": dict(presentations=[{}] * 70, condition_provenance_stale_frames=7,
                       peak_queued_frames=9, peak_resident_chunks=1, peak_ahead_chunks=0),
            "s1": dict(presentations=[{}] * 30, condition_provenance_stale_frames=0,
                       peak_queued_frames=20, peak_resident_chunks=2, peak_ahead_chunks=1)},
            discarded_work={"s0": dict(fully_flushed_denoise_context_s=2.),
                            "s1": dict(fully_flushed_denoise_context_s=3.)})
        self.put("direction3/worker-update.json", dict(samples=[sample]))
        result = self.direction(3)
        arm = result["metrics"]["update"]["arms"][0]
        self.assertAlmostEqual(arm["condition_provenance_stale_fraction"], .07)
        self.assertEqual(arm["buffer_peaks"], dict(peak_queued_frames=20,
                         peak_resident_chunks=2, peak_ahead_chunks=1))
        self.assertEqual(arm["fully_flushed_denoise_context_wall_s"], 5.)
        self.assertEqual(result["metrics"]["profile"]["effective_warmup_blocks"], 4)
        self.assertEqual(result["outcome"]["label"], UNVERIFIED)
        rendered = render_markdown(self.build())
        self.assertIn("从 0 开始", rendered)
        self.assertIn("当前播放块之外", rendered)
        sample["playback"]["s1"].pop("condition_provenance_stale_frames")
        sample.pop("discarded_work")
        self.put("direction3/worker-update.json", dict(samples=[sample]))
        arm = self.direction(3)["metrics"]["update"]["arms"][0]
        self.assertIsNone(arm["condition_provenance_stale_fraction"])
        self.assertIsNone(arm["fully_flushed_denoise_context_wall_s"])

    def test_media_and_failure_evidence_are_linked(self):
        self.put("media/manifest.json", dict(status="incomplete", errors=["missing output"]))
        self.put("direction3/update/blind-windows/public/manifest.json", dict(items=[]))
        self.put("direction3/update/blind-windows/private-key.json", dict(secret="must not link"))
        log = self.root / "direction1/correctness.log"
        log.parent.mkdir()
        log.write_text("failure")
        result = self.build()
        rendered = render_markdown(result)
        self.assertIn(str(log), rendered)
        self.assertIn(str(self.root / "media/manifest.json"), rendered)
        self.assertNotIn("private-key", rendered)

    def test_corrupt_supervisor_cannot_fall_back_to_success(self):
        self.chain(1, "online")
        path = self.put("direction1/supervisor-online.json", {})
        path.write_text("{")
        self.put("direction1/worker-online.json", dict(status="passed", eligible=True, rows=[]))
        result = self.build()
        self.assertEqual(result["directions"][0]["outcome"]["label"], UNVERIFIED)
        self.assertEqual(len(result["evidence_errors"]), 1)
        self.assertIn("读取失败", render_markdown(result))

    def test_write_report_is_repeatable_and_does_not_mutate_inputs(self):
        evidence = self.supervisor(1, "profile", rows=[])
        before = evidence.read_bytes()
        first = write_report(self.root, self.ledger, PHASES)
        second = write_report(self.root, self.ledger, PHASES)
        self.assertEqual(first, second)
        self.assertEqual(evidence.read_bytes(), before)
        self.assertTrue((self.root / "REPORT.md").is_file())
        self.assertEqual(json.loads((self.root / "report.json").read_text())["schema_version"], "ditserve.directions.report.v1")

    def test_corrupt_ledger_is_unknown_not_zero_spend(self):
        self.ledger.write_text("{")
        result = self.build()
        self.assertEqual(result["budget_summary"]["status"], "unavailable")
        self.assertIn("费用未知", render_markdown(result))


class BudgetSummaryTests(unittest.TestCase):
    def test_settled_and_active_phase_time_included_without_changing_ledger(self):
        ledger = dict(used_gpu_seconds=30., limit_gpu_seconds=86400.,
            stage_limits_gpu_seconds=dict(direction1=28800., reserve=7200.),
            stage_used_gpu_seconds=dict(direction1=30., reserve=0.),
            sessions=[dict(stage="direction1", gpu_seconds=30., identity=dict(direction=1, phase="profile"))],
            active=dict(stage="direction1", started_monotonic=80., gpu_count=2,
                        identity=dict(direction=1, phase="correctness")))
        before = json.dumps(ledger, sort_keys=True)
        result = budget_summary(ledger, now_monotonic=100.)
        self.assertEqual(result["status"], "unsettled")
        self.assertEqual(result["including_active_gpu_seconds"], 70.)
        self.assertEqual(result["provisional_gpu_seconds"], 40.)
        self.assertEqual(result["stages"][0]["including_active_gpu_seconds"], 70.)
        self.assertEqual(result["phases"][1]["provisional_gpu_seconds"], 40.)
        self.assertTrue(result["phases"][1]["unsettled"])
        self.assertEqual(json.dumps(ledger, sort_keys=True), before)

    def test_reserve_is_charged_to_reserve_and_keeps_repaired_phase_identity(self):
        result = budget_summary(dict(used_gpu_seconds=4., stage_used_gpu_seconds=dict(direction2=0., reserve=4.),
            sessions=[dict(stage="reserve", gpu_seconds=4., identity=dict(direction=2, phase="pressure"))]))
        self.assertEqual(result["phases"][0]["stage"], "reserve")
        self.assertEqual(result["phases"][0]["phase"], "pressure")
        self.assertEqual(result["status"], "settled")

    def test_unknown_active_clock_never_claims_zero_or_settled(self):
        result = budget_summary(dict(used_gpu_seconds=10., active=dict(started_monotonic=200., gpu_count=1)),
                                now_monotonic=100.)
        self.assertEqual(result["status"], "unsettled")
        self.assertIsNone(result["including_active_gpu_seconds"])
        self.assertIsNone(result["provisional_gpu_seconds"])


if __name__ == "__main__":
    unittest.main()
