"""Frozen workloads preserve arrivals and do not hide a changed load."""

import copy
import unittest

from ditserve.directions.direction1 import frozen_requests
from ditserve.scheduling.workload import SCENARIOS, make_trace


class SchedulingWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(prompts={"video": ["v0", "v1"], "image": ["i0", "i1"]},
                           shape=dict(height=480, width=832, frames=81))
        self.costs = dict(video_service_ms=1000., image_service_ms=10.)

    def test_originals_preserve_old_trace_arrivals_seeds_and_deadlines(self):
        for name, load, pattern in (("original_burst", .9, "burst"), ("original_ordinary", .5, "ordinary")):
            rows = make_trace(self.config, self.costs, name, 301)
            old = frozen_requests(count=24, seed=301, load=load, pattern=pattern, video_ms=1000., image_ms=10.)
            for actual, reference in zip(rows, old):
                self.assertEqual({key: actual[key] for key in reference}, reference)

    def test_contention_preserves_burst_arrivals_while_adding_video_work(self):
        base = make_trace(self.config, self.costs, "original_burst", 301)
        added = make_trace(self.config, self.costs, "video_contention", 301)
        self.assertEqual([r["arrival_ms"] for r in base], [r["arrival_ms"] for r in added])
        self.assertEqual([r["request_id"] for r in added if r["kind"] == "video"], ["r000", "r006", "r012"])
        self.assertGreater(sum(r["service_ms"] for r in added), sum(r["service_ms"] for r in base))
        for row in added:
            multiplier = 1.5 if row["kind"] == "video" else 2.5
            self.assertEqual(row["deadline_ms"], row["arrival_ms"] + multiplier * row["service_ms"])

    def test_tight_trace_changes_only_video_deadline_factor(self):
        rows = make_trace(self.config, self.costs, "tight_deadline", 401)
        old = frozen_requests(count=24, seed=401, load=.9, pattern="ordinary", video_ms=1000., image_ms=10.)
        for row, reference in zip(rows, old):
            self.assertEqual(row["arrival_ms"], reference["arrival_ms"])
            expected = 1.25 if row["kind"] == "video" else 2.5
            self.assertEqual(row["deadline_ms"], row["arrival_ms"] + expected * row["service_ms"])

    def test_prompts_alternate_separately_by_modality_and_shape_is_not_shared(self):
        rows = make_trace(self.config, self.costs, "video_contention", 401)
        for kind, prefix in (("video", "v"), ("image", "i")):
            selected = [r for r in rows if r["kind"] == kind]
            self.assertEqual([r["prompt"] for r in selected], [prefix + str(i % 2) for i in range(len(selected))])
        rows[0]["shape"]["frames"] = 1
        self.assertEqual(rows[1]["shape"]["frames"], 81)
        self.assertEqual(self.config["shape"]["frames"], 81)

    def test_reproducible_and_independent_for_each_arm_without_config_mutation(self):
        original = copy.deepcopy(self.config)
        for scenario in SCENARIOS:
            first = make_trace(self.config, self.costs, scenario, 402)
            second = make_trace(self.config, self.costs, scenario, 402)
            self.assertEqual(first, second)
            first[0]["status"] = "completed"
            self.assertEqual(second[0]["status"], "pending")
            self.assertEqual(len(first), 24)
        self.assertEqual(self.config, original)

    def test_invalid_scenario_and_service_do_not_produce_a_trace(self):
        with self.assertRaises(ValueError):
            make_trace(self.config, self.costs, "unknown", 301)
        with self.assertRaises(ValueError):
            make_trace(self.config, dict(self.costs, video_service_ms=float("nan")), "original_burst", 301)


if __name__ == "__main__":
    unittest.main()
