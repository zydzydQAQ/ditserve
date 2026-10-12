import copy
import math
import statistics
import unittest

from ditserve.benchmark.common import seal
from ditserve.benchmark.workload import CORE_FIELDS, PATTERNS, build_trace, to_core_trace, validate_trace


def fixtures():
    config = {"model": {"id": "fixture-model", "revision": "a" * 40},
              "shapes": {"short": {"height": 256, "width": 256, "frames": 9, "steps": 32, "weight": 1},
                         "long": {"height": 256, "width": 256, "frames": 17, "steps": 32, "weight": 1}},
              "count": 60, "slo_scale": 5, "cache": {"enabled": True, "threshold": .08, "max_continuous": 3},
              "guidance_scale": 5}
    dataset = seal({"schema_version": "ditserve.benchmark.dataset.v1", "kind": "vbench", "source": {},
                    "samples": [{"sample_id": "s%d" % index, "prompt": "prompt %d" % index,
                                 "split": "evaluation", "service_split": "dev" if index == 0 else "test",
                                 "timestamp": index * 10} for index in range(10)]})
    profile = seal({"schema_version": "ditserve.benchmark.profile.v1", "backend": "synthetic",
                    "measurement_kind": "synthetic", "config": copy.deepcopy(config), "shapes": {
                        key: dict(shape, service_ms=100 if key == "short" else 200,
                                  prepare_ms=10, step_ms=[2] * shape["steps"], decode_ms=10,
                                  peak_increment_mb=8, state_mb=4) for key, shape in config["shapes"].items()}})
    return config, dataset, profile


def contents(trace):
    return [{key: value for key, value in row.items() if key not in ("arrival_ms", "deadline_ms")}
            for row in trace["requests"]]


class BenchmarkWorkloadTests(unittest.TestCase):
    def test_rate_change_only_scales_arrivals_not_request_content_or_deadline_budget(self):
        fixtures_ = fixtures()
        for pattern in ("poisson", "gamma", "batch_burst", "foreground_burst"):
            with self.subTest(pattern=pattern):
                slow = build_trace(*fixtures_, pattern=pattern, rate_rps=2, seed=17)
                fast = build_trace(*fixtures_, pattern=pattern, rate_rps=4, seed=17)
                self.assertEqual(contents(slow), contents(fast))
                self.assertEqual(slow, build_trace(*fixtures_, pattern=pattern, rate_rps=2, seed=17))
                for first, second in zip(slow["requests"], fast["requests"]):
                    self.assertAlmostEqual(first["arrival_ms"], second["arrival_ms"] * 2)
                    self.assertAlmostEqual(first["deadline_ms"] - first["arrival_ms"],
                                           second["deadline_ms"] - second["arrival_ms"])

    def test_arrival_patterns_are_explicit_and_content_streams_independent(self):
        fixtures_ = fixtures()
        reference = build_trace(*fixtures_, pattern="poisson")
        for pattern in ("gamma", "batch_burst", "offline_all"):
            trace = build_trace(*fixtures_, pattern=pattern)
            self.assertEqual(contents(reference), contents(trace))
        offline = build_trace(*fixtures_, pattern="offline_all")
        self.assertEqual({row["arrival_ms"] for row in offline["requests"]}, {0})
        burst = build_trace(*fixtures_, pattern="batch_burst")
        times = [row["arrival_ms"] for row in burst["requests"]]
        for index in range(0, len(times), 3):
            self.assertEqual(len(set(times[index:index + 3])), 1)
        config, dataset, profile = fixtures_
        altered = copy.deepcopy(config)
        altered["shapes"]["long"]["weight"] = 9
        changed = build_trace(altered, dataset, profile, rate_rps=2)
        fixed = build_trace(config, dataset, profile, rate_rps=2)
        self.assertEqual([row["arrival_ms"] for row in changed["requests"]],
                         [row["arrival_ms"] for row in fixed["requests"]])
        self.assertEqual([(row["prompt"], row["seed"]) for row in changed["requests"]],
                         [(row["prompt"], row["seed"]) for row in fixed["requests"]])

    def test_gamma_mean_and_coefficient_of_variation(self):
        for cv in (1, 2):
            trace = build_trace(*fixtures(), pattern="gamma", rate_rps=2, cv=cv, count=12000)
            times = [0] + [row["arrival_ms"] for row in trace["requests"]]
            gaps = [b - a for a, b in zip(times, times[1:])]
            self.assertLess(abs(statistics.mean(gaps) / 500 - 1), .08)
            self.assertLess(abs(statistics.pstdev(gaps) / statistics.mean(gaps) / cv - 1), .10)

    def test_foreground_burst_is_one_long_then_three_short_with_truncated_cycle(self):
        trace = build_trace(*fixtures(), pattern="foreground_burst", rate_rps=2, count=6)
        self.assertEqual([row["arrival_ms"] for row in trace["requests"]], [0, 500, 500, 500, 2000, 2500])
        self.assertEqual([row["shape_id"] for row in trace["requests"]], ["long", "short", "short", "short", "long", "short"])
        loaded = build_trace(*fixtures(), pattern="foreground_burst", load=1, count=6)
        self.assertEqual(loaded["arrival"]["mean_standalone_service_ms"], 125)
        self.assertEqual(loaded["arrival"]["rate_rps"], 8)

    def test_mapping_order_does_not_change_identical_frozen_config(self):
        config, dataset, profile = fixtures()
        reordered = copy.deepcopy(config)
        reordered["shapes"] = dict(reversed(list(config["shapes"].items())))
        self.assertEqual(build_trace(config, dataset, profile), build_trace(reordered, dataset, profile))

    def test_native_vidprom_timestamps_have_explicit_timezone_assumption(self):
        replay = [{"timestamp": "Sun Sep 3 12:27:44 2023"}, {"timestamp": "Sun Sep 3 12:27:46 2023"}]
        trace = build_trace(*fixtures(), pattern="replay", count=2, replay=replay)
        self.assertEqual([row["arrival_ms"] for row in trace["requests"]], [0, 2000])
        self.assertEqual(trace["arrival"]["replay_timezone"], "UTC")
        self.assertIn("Assume", trace["arrival"]["naive_timestamp_semantics"])

    def test_replay_sorts_stably_scales_times_and_uses_named_heldout_samples(self):
        replay = [{"arrival_ms": 300, "sample_id": "s3"}, {"arrival_ms": 100, "sample_id": "s1"},
                  {"arrival_ms": 100, "sample_id": "s2"}]
        trace = build_trace(*fixtures(), pattern="replay", count=3, replay=replay, time_scale=2)
        self.assertEqual([row["arrival_ms"] for row in trace["requests"]], [0, 0, 400])
        self.assertEqual([row["sample_id"] for row in trace["requests"]], ["s1", "s2", "s3"])
        self.assertIsNotNone(trace["arrival"]["replay_sha256"])
        with self.assertRaises(ValueError):
            build_trace(*fixtures(), pattern="replay", count=1, replay=[{"arrival_ms": 0, "sample_id": "s0"}])
        collection = build_trace(*fixtures(), pattern="replay", count=2)
        self.assertEqual([row["arrival_ms"] for row in collection["requests"]], [0, 10000])
        self.assertEqual(collection["arrival"]["replay_source"], "dataset_collection_timestamps")

    def test_hash_and_strict_adapter(self):
        trace = build_trace(*fixtures())
        core = to_core_trace(trace)
        self.assertTrue(all(set(row) == set(CORE_FIELDS) for row in core))
        self.assertNotIn("prompt", core[0])
        tampered = copy.deepcopy(trace)
        tampered["requests"][0]["prompt"] = "changed"
        with self.assertRaises(ValueError):
            validate_trace(tampered)
        duplicate = copy.deepcopy(trace)
        duplicate["requests"][1]["request_id"] = duplicate["requests"][0]["request_id"]
        with self.assertRaises(ValueError):
            validate_trace(seal(duplicate))
        bad_budget = copy.deepcopy(trace)
        bad_budget["requests"][0]["deadline_ms"] += 1
        with self.assertRaises(ValueError):
            validate_trace(seal(bad_budget))

    def test_invalid_parameters_and_profile_shape_mismatch(self):
        for options in ({"rate_rps": 0}, {"rate_rps": float("nan")}, {"cv": 0}, {"count": True},
                        {"load": 0}, {"time_scale": 2}, {"pattern": "bursty"},
                        {"pattern": "offline_all", "rate_rps": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_trace(*fixtures(), **options)
        config, dataset, profile = fixtures()
        profile["shapes"]["short"]["frames"] = 99
        with self.assertRaises(ValueError):
            build_trace(config, dataset, seal(profile))

    def test_profile_and_request_generation_settings_cannot_drift(self):
        config, dataset, profile = fixtures()
        for update in ({"guidance_scale": 1}, {"negative_prompt": "different"},
                       {"cache": {"enabled": False, "threshold": .08, "max_continuous": 3}}):
            changed = dict(config, **update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                build_trace(changed, dataset, profile)
        original = build_trace(config, dataset, profile)
        for field, value in (("cache_threshold", .12), ("cache_enabled", False),
                             ("guidance_scale", 1), ("negative_prompt", "changed"), ("max_continuous", 0)):
            changed = copy.deepcopy(original)
            changed["requests"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_trace(seal(changed))


if __name__ == "__main__":
    unittest.main()
