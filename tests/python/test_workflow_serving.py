"""Arrival/queue/measurement contracts run without torch or CUDA."""

import copy
from contextlib import ExitStack
import heapq
import json
import unittest
from unittest.mock import patch

from ditserve.evaluation.workflow_serving import (
    ArrivalScheduler, MANIFEST_SCHEMA, ProcessPipelines, _decorate_rows,
    memory_margin_gate, run_observation, summarize_requests, validate_manifest,
)


def request(identifier="r0", arrival=0, frames=17):
    return dict(request_id=identifier, arrival_ms=arrival, slo_ms=100,
                spec=dict(request_id=identifier, prompt="A red boat.", negative_prompt="", seed=7,
                          height=480, width=832, frames=frames, steps=20, guidance_scale=5.))


def manifest():
    return dict(schema=MANIFEST_SCHEMA, kind="online", placement="dual_cpu", policy="shared_fifo",
                model_path="/models/wan", observations=[dict(id="t0", requests=[request()])])


class ManifestTests(unittest.TestCase):
    def test_defaults_and_normalization_do_not_mutate_frozen_input(self):
        value = manifest()
        original = copy.deepcopy(value)
        checked = validate_manifest(value)
        self.assertEqual(value, original)
        self.assertEqual(checked["controller_cpu"], 59)
        self.assertEqual(len(set(sum(checked["worker_cpu_sets"], []))), 16)
        self.assertEqual(checked["prefetch_depth"], 1)
        self.assertEqual(checked["queue_limit"], 64)
        self.assertFalse(checked["save_videos"])
        json.dumps(checked, allow_nan=False)

    def test_save_videos_requires_an_explicit_boolean(self):
        value = manifest()
        value["save_videos"] = True
        self.assertTrue(validate_manifest(value)["save_videos"])
        for invalid in (1, 0, "true", None):
            value["save_videos"] = invalid
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "save_videos"):
                validate_manifest(value)

    def test_worker_and_controller_affinity_must_be_disjoint(self):
        value = manifest()
        value["worker_cpu_sets"] = [list(range(8)), list(range(8))]
        with self.assertRaises(ValueError):
            validate_manifest(value)
        value = manifest()
        value["controller_cpu"] = 0
        with self.assertRaises(ValueError):
            validate_manifest(value)

    def test_invalid_trace_and_missing_cost_are_rejected_before_start(self):
        for change in ("duplicate", "early", "nan", "cost", "mismatch"):
            with self.subTest(change=change):
                value = manifest()
                rows = value["observations"][0]["requests"]
                if change == "duplicate":
                    rows.append(request())
                elif change == "early":
                    rows[:] = [request("r0", 2), request("r1", 1)]
                elif change == "nan":
                    rows[0]["slo_ms"] = float("nan")
                elif change == "cost":
                    value["policy"] = "least_work_fifo"
                else:
                    rows[0]["spec"]["request_id"] = "different"
                with self.assertRaises(ValueError):
                    validate_manifest(value)

    def test_frozen_frame_costs_are_accepted_for_both_classes(self):
        value = manifest()
        value.update(policy="least_work_fifo", estimates_ms={"17": dict(encode=10, gpu=20), "33": dict(encode=10, gpu=40)})
        value["observations"][0]["requests"].append(request("r1", frames=33))
        self.assertEqual(validate_manifest(value)["observations"][0]["estimates_ms"], value["estimates_ms"])


class SchedulerTests(unittest.TestCase):
    def test_no_release_before_arrival_or_duplicate(self):
        scheduler = ArrivalScheduler("shared_fifo", [0, 1])
        with self.assertRaises(ValueError):
            scheduler.arrive(request(), 10, 9)
        scheduler.arrive(request(), 10, 10)
        with self.assertRaises(ValueError):
            scheduler.arrive(request(), 10, 11)

    def test_fifo_depth_and_idle_gpu_priority(self):
        scheduler = ArrivalScheduler("shared_fifo", [0, 1])
        for index in range(6):
            scheduler.arrive(request("r%d" % index), 0, 0)
        dispatched = scheduler.dispatch(0)
        self.assertEqual([(worker, row["request_id"]) for worker, row in dispatched], [(0, "r0"), (1, "r1")])
        scheduler.event(dict(kind="gpu_started", request_id="r0", gpu_start_ns=1))
        scheduler.event(dict(kind="gpu_started", request_id="r1", gpu_start_ns=1))
        self.assertEqual([(worker, row["request_id"]) for worker, row in scheduler.dispatch(1)], [(0, "r2"), (1, "r3")])
        self.assertEqual(list(scheduler.shared), ["r4", "r5"])
        self.assertTrue(all(len(rows) == 2 for rows in scheduler.active.values()))
        scheduler.event(dict(kind="completed", request_id="r1", complete_ns=2))
        self.assertFalse(scheduler.dispatch(2))  # No second lookahead before r3 takes the GPU.
        scheduler.event(dict(kind="gpu_started", request_id="r3", gpu_start_ns=3))
        self.assertEqual(scheduler.dispatch(3)[0][1]["request_id"], "r4")
        self.assertEqual(scheduler.rows["r4"]["assigned_worker"], 1)

    def test_least_work_deducts_service_not_waiting(self):
        scheduler = ArrivalScheduler("least_work_fifo", [0, 1], {"17": dict(encode=10, gpu=20)})
        scheduler.arrive(request("r0"), 0, 0)
        scheduler.dispatch(0)
        self.assertEqual(scheduler.remaining_ms(0, 100_000_000), 30)
        scheduler.event(dict(kind="encode_started", request_id="r0", encode_start_ns=100_000_000))
        self.assertEqual(scheduler.remaining_ms(0, 105_000_000), 25)
        scheduler.event(dict(kind="encoded", request_id="r0", encode_start_ns=100_000_000, encode_end_ns=110_000_000))
        self.assertEqual(scheduler.remaining_ms(0, 200_000_000), 20)
        scheduler.event(dict(kind="gpu_started", request_id="r0", gpu_start_ns=200_000_000))
        self.assertEqual(scheduler.remaining_ms(0, 205_000_000), 15)
        self.assertEqual(scheduler.remaining_ms(0, 300_000_000), 0)

    def test_least_work_waits_for_available_slot_and_retains_global_fifo(self):
        scheduler = ArrivalScheduler("least_work_fifo", [0, 1], {"17": dict(encode=10, gpu=20), "33": dict(encode=10, gpu=60)})
        for row in (request("long", frames=33), request("short"), request("next")):
            scheduler.arrive(row, 0, 0)
        self.assertEqual(list(scheduler.shared), ["long", "short", "next"])
        self.assertTrue(all("assigned_worker" not in row for row in scheduler.rows.values()))
        assigned = scheduler.dispatch(0)
        self.assertEqual([row["request_id"] for worker, row in assigned if worker == 1], ["short"])
        scheduler.event(dict(kind="gpu_started", request_id="short", gpu_start_ns=1))
        self.assertEqual(scheduler.dispatch(1)[0][1]["request_id"], "next")

    def test_least_work_selects_lower_remaining_cost_among_free_lookaheads(self):
        scheduler = ArrivalScheduler("least_work_fifo", [0, 1], {"17": dict(encode=10, gpu=20), "33": dict(encode=10, gpu=60)})
        for row in (request("long", frames=33), request("short"), request("next")):
            scheduler.arrive(row, 0, 0)
        scheduler.dispatch(0)
        for identifier in ("long", "short"):
            scheduler.event(dict(kind="encoded", request_id=identifier, encode_start_ns=0, encode_end_ns=1))
            scheduler.event(dict(kind="gpu_started", request_id=identifier, gpu_start_ns=1))
        worker, row = scheduler.dispatch(2)[0]
        self.assertEqual((worker, row["request_id"]), (1, "next"))
        self.assertEqual(row["route_decision_ns"], 2)

    def test_waiting_queue_is_bounded_and_inservice_does_not_count(self):
        scheduler = ArrivalScheduler("shared_fifo", [0], queue_limit=2)
        scheduler.arrive(request("r0"), 0, 0)
        scheduler.dispatch(0)
        scheduler.event(dict(kind="encode_started", request_id="r0", encode_start_ns=0))
        self.assertTrue(scheduler.arrive(request("r1"), 0, 1))
        self.assertTrue(scheduler.arrive(request("r2"), 0, 1))
        self.assertFalse(scheduler.arrive(request("r3"), 0, 1))
        self.assertEqual(scheduler.rows["r3"]["status"], "rejected")

    def test_dedicated_cross_process_status_order_cannot_regress_completion(self):
        scheduler = ArrivalScheduler("shared_fifo", [1])
        scheduler.arrive(request(), 10, 10)
        scheduler.dispatch(10)
        scheduler.event(dict(kind="gpu_started", request_id="r0", gpu_start_ns=30))
        scheduler.event(dict(kind="encoded", request_id="r0", encode_start_ns=11, encode_end_ns=20))
        self.assertEqual(scheduler.rows["r0"]["status"], "gpu")
        scheduler.event(dict(kind="completed", request_id="r0", complete_ns=40))
        scheduler.event(dict(kind="encode_started", request_id="r0", encode_start_ns=11))
        self.assertEqual(scheduler.rows["r0"]["status"], "completed")
        self.assertFalse(scheduler.active[1])

    def test_isolated_idle_ties_rotate_between_the_two_workers(self):
        scheduler = ArrivalScheduler("shared_fifo", [0, 1], rotate_idle=True)
        assigned = []
        for index in range(4):
            identifier = "r%d" % index
            scheduler.arrive(request(identifier), index * 10, index * 10)
            assigned.append(scheduler.dispatch(index * 10)[0][0])
            scheduler.event(dict(kind="completed", request_id=identifier, complete_ns=index * 10 + 1))
        self.assertEqual(assigned, [0, 1, 0, 1])

    def test_shared_fifo_ties_use_gpu_number_and_predictions_are_frozen(self):
        scheduler = ArrivalScheduler("shared_fifo", [0, 1], {"17": dict(encode=10, gpu=20)})
        scheduler.arrive(request(), 0, 0)
        self.assertEqual(scheduler.dispatch(0)[0][0], 0)
        self.assertEqual(scheduler.rows["r0"]["predicted_complete_ns"], 30_000_000)
        scheduler.event(dict(kind="completed", request_id="r0", complete_ns=40_000_000))
        row = _decorate_rows([scheduler.rows["r0"]], 0)[0]
        self.assertEqual(row["prediction_error_ms"], 10)
        scheduler.arrive(request("r1"), 40_000_000, 40_000_000)
        self.assertEqual(scheduler.dispatch(40_000_000)[0][0], 0)


class MetricTests(unittest.TestCase):
    def test_arrival_latency_and_failed_denominator_with_drain(self):
        origin = 1_000_000_000
        first = dict(request("r0"), status="completed", scheduled_arrival_ns=origin + 10_000_000,
                     encode_start_ns=origin + 20_000_000, complete_ns=origin + 60_000_000)
        second = dict(request("r1"), status="rejected", scheduled_arrival_ns=origin + 20_000_000)
        third = dict(request("r2"), status="queued", scheduled_arrival_ns=origin + 30_000_000)
        summary = summarize_requests([first, second, third], origin, origin + 200_000_000)
        self.assertEqual(summary["latency_p50_ms"], 50)
        self.assertEqual(summary["slo_attainment"], 1 / 3)
        self.assertEqual(summary["throughput_per_second"], 5)
        self.assertEqual(summary["arrival_window_ms"], 20)
        self.assertEqual(summary["unfinished"], 1)
        self.assertIsNone(summary["qualified_goodput_per_second"])
        decorated = _decorate_rows([first], origin)[0]
        self.assertEqual(decorated["queue_wait_ms"], 10)
        self.assertEqual(decorated["response_ms"], 50)
        self.assertEqual(decorated["timestamps_ms"]["complete"], 60)

    def test_empty_completion_is_missing_latency_not_zero(self):
        row = dict(request(), status="not_arrived", scheduled_arrival_ns=None)
        summary = summarize_requests([row], 0, 1)
        self.assertIsNone(summary["latency_p95_ms"])
        self.assertEqual(summary["slo_met"], 0)

    def test_frozen_arrival_window_has_a_separate_rate(self):
        row = dict(request(), status="completed", scheduled_arrival_ns=0, complete_ns=10_000_000)
        summary = summarize_requests([row], 0, 20_000_000, arrival_window_ms=100)
        self.assertEqual(summary["throughput_per_second"], 50)
        self.assertEqual(summary["ontime_per_arrival_window_second"], 10)
        self.assertEqual(summary["arrival_window_ms"], 100)

    def test_memory_margin_is_minimum_not_average_and_missing_is_unverified(self):
        gib = 1024 ** 3
        rows = [dict(allocator=dict(device="cuda:%d" % index, device_free_bytes=2 * gib, device_total_bytes=24 * gib)) for index in (0, 1)]
        self.assertEqual(memory_margin_gate(rows)["status"], "passed")
        self.assertEqual(memory_margin_gate(rows[:1])["status"], "unverified")
        samples = [dict(device=index, timestamp_ns=5, memory_free_bytes=2 * gib, memory_total_bytes=24 * gib) for index in (0, 1)]
        samples.append(dict(device=1, timestamp_ns=6, memory_free_bytes=gib, memory_total_bytes=24 * gib))
        samples.append(dict(device=0, timestamp_ns=100, memory_free_bytes=0, memory_total_bytes=24 * gib))
        self.assertEqual(memory_margin_gate(samples, sampled=True, origin_ns=0, end_ns=10)["status"], "failed")

    def test_slow_model_loading_none_poll_does_not_fail_startup(self):
        class Process:
            def start(self):
                pass
        pipelines = object.__new__(ProcessPipelines)
        pipelines.configuration = dict(startup_timeout_seconds=1)
        pipelines.processes, pipelines.ready = [Process(), Process()], []
        with patch.object(pipelines, "get", side_effect=[None, dict(kind="ready", process="p0"), None, dict(kind="ready", process="p1")]):
            pipelines.start()
        self.assertEqual(len(pipelines.ready), 2)


class FakeClock:
    def __init__(self):
        self.now = 1_000_000_000

    def ns(self):
        return self.now

    def seconds(self):
        return self.now / 1e9


class FakePipelines:
    workers = [0, 1]
    ready = []

    def __init__(self, clock, duration_ms=2):
        self.clock, self.duration_ns = clock, round(duration_ms * 1e6)
        self.sent, self.events, self.closed = [], [], False
        self.sequence = 0
        self.gpu_free = {0: 0, 1: 0}
        self.free_bytes = 2 * 1024 ** 3
        self.fail_after_ns = None

    def control(self, op, observation):
        return [dict(process="process%d" % index, cpu_core_seconds=.01, process_cpu_seconds=0,
                     allocator=dict(device="cuda:%d" % index, device_free_bytes=self.free_bytes,
                                    device_total_bytes=24 * 1024 ** 3)) for index in (0, 1)]

    def send(self, worker, row):
        self.sent.append((self.clock.now, worker, copy.deepcopy(row)))
        start = self.clock.now + 100_000
        encoded = start + 1_000_000
        gpu = max(encoded, self.gpu_free[worker])
        complete = gpu + self.duration_ns
        self.gpu_free[worker] = complete
        for when, event in ((start, dict(kind="encode_started", encode_start_ns=start)),
                            (encoded, dict(kind="encoded", encode_start_ns=start, encode_end_ns=encoded)),
                            (gpu, dict(kind="gpu_started", gpu_start_ns=gpu)),
                            (complete, dict(kind="completed", encode_start_ns=start, encode_end_ns=encoded,
                                            gpu_start_ns=gpu, complete_ns=complete))):
            self.sequence += 1
            heapq.heappush(self.events, (when, self.sequence, dict(event, request_id=row["request_id"])))

    def get(self, deadline=None, timeout=.05):
        if self.fail_after_ns is not None and self.clock.now >= self.fail_after_ns:
            raise RuntimeError("fake worker failed at cutoff")
        until = self.clock.now + round(timeout * 1e9)
        if self.events and self.events[0][0] <= until:
            when, unused, event = heapq.heappop(self.events)
            self.clock.now = max(self.clock.now, when)
            return event
        self.clock.now = until
        return None

    def close(self):
        self.closed = True


class ObservationLifecycleTests(unittest.TestCase):
    def run_fake(self, observation, duration_ms=2, free_bytes=None, fail_at_cutoff=False):
        clock = FakeClock()
        pipelines = FakePipelines(clock, duration_ms)
        if free_bytes is not None:
            pipelines.free_bytes = free_bytes
        if fail_at_cutoff:
            pipelines.fail_after_ns = clock.now + round(observation["observation_cutoff_ms"] * 1e6)
        pipelines.sampler_stopped = False
        class Sampler:
            def __init__(self, **kwargs):
                self.samples = []

            def start(self):
                self.samples = [dict(device=index, timestamp_ns=clock.now, memory_free_bytes=2 * 1024 ** 3,
                                     memory_total_bytes=24 * 1024 ** 3) for index in (0, 1)]

            def stop(self):
                pipelines.sampler_stopped = True

            def summary(self, start, end):
                return dict(devices={}, errors=[])
        value = manifest()
        value["observations"] = [observation]
        configuration = validate_manifest(value)
        snapshots = []
        with ExitStack() as stack:
            stack.enter_context(patch("ditserve.evaluation.workflow_serving.time.perf_counter_ns", clock.ns))
            stack.enter_context(patch("ditserve.evaluation.workflow_serving.time.monotonic", clock.seconds))
            stack.enter_context(patch("ditserve.evaluation.workflow_serving.time.process_time_ns", return_value=0))
            stack.enter_context(patch("ditserve.evaluation.workflow_metrics.NvmlSampler", Sampler))
            result = run_observation(configuration, configuration["observations"][0], pipelines,
                                     lambda row: snapshots.append(copy.deepcopy(row)))
        return result, pipelines, snapshots

    def test_controller_releases_online_jobs_after_scheduled_arrival(self):
        observation = dict(id="online", role="measurement", arrival_window_ms=30,
                           requests=[request("r0"), request("r1", 20)])
        result, pipelines, snapshots = self.run_fake(observation)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["performance_evidence"])
        self.assertEqual(result["summary"]["completed"], 2)
        self.assertEqual(result["summary"]["arrival_window_ms"], 30)
        for sent_ns, worker, row in pipelines.sent:
            self.assertGreaterEqual(sent_ns, row["scheduled_arrival_ns"])
        self.assertAlmostEqual(result["requests"][1]["latency_ms"], 3.1)
        self.assertEqual(snapshots[-1]["status"], "completed")

    def test_isolated_arrivals_follow_prior_completion_and_rotate_workers(self):
        observation = dict(id="isolated", role="isolated", requests=[request("r0"), request("r1"), request("r2")])
        result, pipelines, unused = self.run_fake(observation)
        self.assertEqual([worker for when, worker, row in pipelines.sent], [0, 1, 0])
        for left, right in zip(result["requests"], result["requests"][1:]):
            self.assertGreaterEqual(right["scheduled_arrival_ns"], left["complete_ns"])
            self.assertAlmostEqual(right["latency_ms"], 3.1)

    def test_frozen_cutoff_stops_own_workers_and_retains_unfinished_denominator(self):
        observation = dict(id="cutoff", role="measurement", arrival_window_ms=30, observation_cutoff_ms=10,
                           requests=[request("r0"), request("r1", 20)])
        result, pipelines, unused = self.run_fake(observation, duration_ms=500)
        self.assertTrue(pipelines.closed)
        self.assertTrue(result["stopped_at_cutoff"])
        self.assertEqual(result["summary"]["wall_ms"], 10)
        self.assertEqual(result["summary"]["unfinished"], 2)
        self.assertEqual(result["summary"]["slo_attainment"], 0)
        self.assertEqual([row["status"] for row in result["requests"]], ["unfinished_at_cutoff", "not_arrived"])
        self.assertEqual(len(pipelines.sent), 1)

    def test_capacity_rejection_never_dispatches_a_measured_request(self):
        result, pipelines, snapshots = self.run_fake(dict(id="small", requests=[request()]), free_bytes=1)
        self.assertEqual(result["status"], "capacity_rejected")
        self.assertFalse(result["performance_evidence"])
        self.assertFalse(pipelines.sent)
        self.assertIsNone(result["summary"]["throughput_per_second"])
        self.assertEqual(snapshots[-1]["capacity_gate"]["status"], "failed")

    def test_failure_during_cutoff_drain_still_cleans_up_and_checkpoints(self):
        observation = dict(id="cutoff", role="measurement", observation_cutoff_ms=10, requests=[request()])
        result, pipelines, snapshots = self.run_fake(observation, duration_ms=500, fail_at_cutoff=True)
        self.assertTrue(pipelines.closed)
        self.assertTrue(pipelines.sampler_stopped)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["performance_evidence"])
        self.assertEqual(snapshots[-1]["summary"]["unfinished"], 1)
        self.assertIn("fake worker failed", snapshots[-1]["cutoff_drain_error"])


if __name__ == "__main__":
    unittest.main()
