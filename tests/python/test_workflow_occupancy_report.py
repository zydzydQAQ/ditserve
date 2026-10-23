"""Occupancy guard traces, numerical gates and prespecified performance rules."""

import copy
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_occupancy_report as audit
from ditserve.evaluation import workflow_occupancy_study as study
from ditserve.evaluation.workflow_serving_study import digest
from test_workflow_dispatch_refresh_report import fixture as dispatch_fixture


def fixture():
    original, job, old_config = dispatch_fixture()
    config = study.native_config(old_config["phases"][0]["profile"], old_config["phases"][0]["workload"])
    observed = [original["phases"][1], original["phases"][2]]
    for phase, actual in zip(config["phases"], observed):
        name = phase["id"]
        actual["id"] = name
        cell = actual["cells"][0]
        cell.update(dispatch_refresh=False, host_origin_monotonic_seconds=1000.)
        durations = [60., 50., 50.] if name == "reference" else [600., 1., 1.]
        events = [dict(start_monotonic_seconds=start, finish_monotonic_seconds=start + duration / 1000,
                       duration_ms=duration, status="exclusive_observed", occupancy_backend=phase["occupancy_backend"],
                       owned_count=4, foreign_count=0,
                       owned_processes=[dict(pid=runtime["pid"], pgid=runtime["pid"],
                           gpu_uuid=f"GPU-{config['workers'][worker]['gpu']}", memory_mib="1024.0")
                           for worker, runtime in cell["worker_runtime"].items()])
                  for start, duration in zip((994., 997., 1000.1), durations)]
        cell["guard_telemetry"] = dict(schema="ditserve.guard-telemetry.v1", occupancy_backend=phase["occupancy_backend"],
            cadence_seconds=2., probe_timeout_seconds=3., fail_closed=True,
            scope="cell startup, warmup and execution; excludes shutdown", heartbeat_calls=100,
            lock_wait_total_ms=.5, lock_wait_max_ms=.1, heartbeat_total_ms=sum(durations)+1.,
            heartbeat_max_ms=max(durations)+.1, probe_calls=3, probe_total_ms=sum(durations),
            probe_max_ms=max(durations), probe_errors=0, events=events, dropped_events=0)
    result = dict(original, config=copy.deepcopy(config), phases=observed)
    for worker in result["config"]["workers"].values():
        worker["gpu_uuid"] = f"GPU-{worker['gpu']}"
    job.update(configuration_sha256=digest(config), metadata=dict(role="occupancy_online", repeat=0))
    return result, job, config


class OccupancyReportTests(unittest.TestCase):
    def test_two_cache_enabled_arms_exact_and_guard_lifecycle(self):
        result, job, config = fixture()
        checked = audit.audit_result(result, job, config)
        self.assertTrue(checked["passed"], checked["errors"])
        self.assertEqual(checked["exact"]["reference/candidate"]["requests"], 24)
        guard = checked["arms"]["candidate"]["guard"]
        self.assertEqual(guard["cold_probe_ms"], 600)
        self.assertEqual(guard["probe_steady_mean_ms"], 1)
        self.assertEqual(guard["warmup_probe_calls"], 2)
        self.assertEqual(guard["online_probe_calls"], 1)
        result["phases"].pop()
        self.assertFalse(audit.audit_result(result, job, config)["passed"])

    def test_guard_controls_counts_status_and_raw_aggregates_fail_closed(self):
        edits = [lambda g: g.update(cadence_seconds=3), lambda g: g.update(probe_timeout_seconds=4),
                 lambda g: g.update(fail_closed=False), lambda g: g.update(probe_calls=True),
                 lambda g: g.update(heartbeat_calls=2), lambda g: g.update(probe_errors=1),
                 lambda g: g.update(dropped_events=1), lambda g: g.update(probe_total_ms=1),
                 lambda g: g["events"].pop(), lambda g: g["events"][0].update(foreign_count=1),
                 lambda g: g["events"][0].update(status="unknown"),
                 lambda g: g["events"][0].update(occupancy_backend="nvidia_smi"),
                 lambda g: g["events"][0].update(duration_ms=0),
                 lambda g: g["events"][1].update(start_monotonic_seconds=994.),
                 lambda g: g.update(heartbeat_total_ms=0., heartbeat_max_ms=0.)]
        for index, edit in enumerate(edits):
            with self.subTest(index=index):
                result, job, config = fixture()
                edit(result["phases"][-1]["cells"][0]["guard_telemetry"])
                self.assertFalse(audit.audit_result(result, job, config)["passed"])

    def test_measured_origin_and_both_guard_intervals_are_required(self):
        for origin in (None, 1., 10000.):
            result, job, config = fixture()
            result["phases"][-1]["cells"][0]["host_origin_monotonic_seconds"] = origin
            self.assertFalse(audit.audit_result(result, job, config)["passed"])

    def test_live_owner_set_and_probe_measurement_bounds_are_checked(self):
        for change in ("empty", "pid", "uuid", "after_end", "resolved_worker_uuid"):
            result, job, config = fixture()
            cell = result["phases"][-1]["cells"][0]
            event = cell["guard_telemetry"]["events"][-1]
            if change == "empty":
                event.update(owned_processes=[], owned_count=0)
            elif change == "pid":
                event["owned_processes"][0]["pid"] += 500
            elif change == "uuid":
                event["owned_processes"][0]["gpu_uuid"] = "GPU-wrong"
            elif change == "after_end":
                start = cell["host_origin_monotonic_seconds"] + cell["elapsed_ms"] / 1000 + 10
                event.update(start_monotonic_seconds=start, finish_monotonic_seconds=start + event["duration_ms"] / 1000)
            else:
                result["config"]["workers"]["wan0"]["gpu_uuid"] = "GPU-wrong"
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)

    def test_configuration_cache_receipt_and_latents_remain_strict(self):
        for change in ("backend", "refresh", "cache", "latent"):
            result, job, config = fixture()
            if change in ("backend", "refresh"):
                config["phases"][-1]["occupancy_backend" if change == "backend" else "dispatch_refresh"] = "nvidia_smi" if change == "backend" else True
                result["config"] = copy.deepcopy(config)
                job["configuration_sha256"] = digest(config)
            else:
                row = next(r for r in result["phases"][-1]["cells"][0]["requests"] if r["kind"] == "video")
                if change == "cache":
                    row["negative_prompt_cache"]["sequence"] += 1
                else:
                    row["outputs"][0]["latent"]["sha256"] = "0" * 64
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)

    def test_screen_excludes_first_cold_probe_and_keeps_separate_safety_limits(self):
        result, _, _ = fixture()
        ev = dict(qualified=True, capacity=True, result=result)
        gate = audit.screening_gate(ev)
        self.assertTrue(gate["passed"], gate)
        self.assertEqual(gate["steady_probe_mean_reduction_percent"], 98.)
        checked = audit.audit_result(result, fixture()[1], fixture()[2])
        arms = {name: dict(audit.report.summary(cell["requests"], cell["elapsed_ms"]),
                          guard=checked["arms"][name]["guard"]) for name, cell in audit.report.native_arms(result).items()}
        for kind, metric, multiplier in (("image", "latency_p95_ms", 1.051), ("video", "latency_p95_ms", 1.051),
                                         ("total", "throughput_per_second", .969), ("guard", "probe_steady_mean_ms", .201)):
            changed = copy.deepcopy(arms)
            changed["candidate"][kind][metric] = changed["reference"][kind][metric] * multiplier
            self.assertFalse(audit.performance_assessment(changed)["passed"], (kind, metric))
        self.assertFalse(audit.screening_gate(dict(ev, qualified=False))["passed"])

    def test_unstarted_confirmations_are_pending_not_numerical_failures(self):
        jobs = [dict(id=study.job_id(seed), decision="occupancy_screening_not_promising",
                     metadata=dict(role="occupancy_online", seed=seed, stage="confirmation")) for seed in study.SEEDS]
        with patch.object(audit.report, "evidence", return_value=dict(launch={}, result={}, qualified=False, capacity=False)), \
                patch.object(audit, "audit_result", side_effect=AssertionError("must not audit unstarted output")):
            summary = audit.summarize(dict(jobs=jobs))
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["promotable"])
        self.assertIsNone(summary["paired_changes_median"])
        self.assertTrue(all(row["status"] == "not_started" and row["audit"] is None for row in summary["per_seed"]))

    def test_promotion_requires_each_seed_safety_and_guard_benefit(self):
        result, job, config = fixture()
        ev = dict(launch=dict(status="completed"), result=result, qualified=True, capacity=True)
        jobs = [dict(job, id=study.job_id(seed), input="fixture", requires=["numeric"],
                     metadata=dict(role="occupancy_online", repeat=0, stage="confirmation", seed=seed)) for seed in study.SEEDS]
        with patch.object(audit.report, "evidence", return_value=ev), \
                patch.object(audit.report, "current_job", return_value={}), \
                patch.object(audit, "read", return_value=config), \
                patch.object(audit.encoding, "raw_capacity", return_value=dict(passed=True)):
            summary = audit.summarize(dict(jobs=jobs))
            self.assertTrue(summary["qualified"])
            self.assertTrue(summary["promotable"])
            self.assertFalse(summary["default_change"])
            self.assertEqual(summary["paired_changes_median"]["reference/candidate"]["guard"]["probe_steady_mean_ms"]["improvement_percent"], 98.)
            with patch.object(audit, "performance_assessment", side_effect=[dict(passed=True), dict(passed=False), dict(passed=True)]):
                failed = audit.summarize(dict(jobs=jobs))
            self.assertTrue(failed["qualified"])
            self.assertFalse(failed["promotable"])


if __name__ == "__main__":
    unittest.main()
