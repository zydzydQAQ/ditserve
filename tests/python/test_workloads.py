import copy
import json
from pathlib import Path
import tempfile
import unittest

from ditserve.evaluation.workloads import (
    build_trace, gpu_service_ms, load_trace, validate_trace, write_trace,
)


def classes():
    return {
        "long": dict(kind="video", service_ms=1000., prepare_ms=0., steps=2,
                     step_compute_ms=400., finish_ms=100., slo_scale=1.5,
                     spec=dict(steps=2, height=704, width=1280, frames=121)),
        "mid": dict(kind="video", service_ms=500., prepare_ms=0., steps=1,
                    step_compute_ms=300., finish_ms=100., slo_scale=1.5,
                    spec=dict(steps=1, height=512, width=896, frames=81)),
        "sana": dict(kind="image", service_ms=10., slo_scale=2.5, spec=dict(steps=2)),
    }


def pool(kind, count=100):
    return dict(kind=kind, source=dict(revision="a" * 40), selection=dict(method="test"),
                samples=[dict(sample_id=f"{kind}-{index}", prompt=f"{kind} prompt {index}")
                         for index in range(count)])


def make(fraction=0.2, pattern="poisson", seed=201, windows=None, quality=0):
    if windows is None:
        windows = [dict(count=20, video_fraction=fraction, pattern=pattern, load=1.0,
                        bursts=2, burst_active_fraction=0.25)]
    return build_trace(classes(), {"image": pool("diffusiondb"), "video": pool("vidprom")},
                       windows, slots=8, seed=seed, quality_per_kind=quality)


class MixedWorkloadTests(unittest.TestCase):
    def test_strict_ratios_and_gpu_time_fraction(self):
        for fraction, expected in ((0.2, 4), (0.5, 10), (0.8, 16)):
            with self.subTest(fraction=fraction):
                trace = make(fraction)
                self.assertTrue(validate_trace(trace))
                manifest = trace["manifest"]
                self.assertEqual(manifest["counts"], {"video": expected, "image": 20 - expected})
                self.assertEqual(manifest["class_counts"]["long"], expected // 2)
                self.assertEqual(manifest["class_counts"]["mid"], expected // 2)
                self.assertAlmostEqual(manifest["normalized_gpu_work"], 1.0)
                gpu = {kind: sum(gpu_service_ms(classes()[row["class"]]) for row in trace["workload"]
                                 if row["kind"] == kind) for kind in ("image", "video")}
                self.assertEqual(manifest["gpu_ms"], gpu)
                total = sum(gpu.values())
                self.assertAlmostEqual(manifest["gpu_time_fraction"]["video"], gpu["video"] / total)
                arrivals = [row["arrival_ms"] for row in trace["workload"]]
                self.assertEqual(arrivals, sorted(arrivals))

    def test_conditional_poisson_uses_independent_arrival_draws(self):
        trace = make(pattern="poisson")
        duration = trace["manifest"]["total_duration_ms"]
        for kind in ("image", "video"):
            arrivals = [row["arrival_ms"] for row in trace["workload"] if row["kind"] == kind]
            self.assertGreater(len(arrivals), 1)
            self.assertEqual(len(arrivals), len(set(arrivals)))
            self.assertTrue(all(0 <= arrival < duration for arrival in arrivals))

    def test_seed_reuse_and_independent_prompt_changes(self):
        first, second = make(seed=201), make(seed=201)
        self.assertEqual(first, second)
        changed = make(seed=202)
        self.assertNotEqual(first["workload"], changed["workload"])
        pools = {"image": pool("diffusiondb"), "video": pool("vidprom")}
        for samples in pools.values():
            for sample in samples["samples"]:
                sample["prompt"] += " alternate"
        alternate = build_trace(classes(), pools, [dict(count=20, video_fraction=0.2, pattern="poisson")],
                                slots=8, seed=201)
        self.assertEqual([row["arrival_ms"] for row in first["workload"]],
                         [row["arrival_ms"] for row in alternate["workload"]])
        self.assertNotEqual([row["spec"]["prompt"] for row in first["workload"]],
                            [row["spec"]["prompt"] for row in alternate["workload"]])
        prompts = [row["spec"]["prompt"] for row in first["workload"]]
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_ratio_t_windows_have_exact_counts_and_ordered_intervals(self):
        windows = [dict(count=20, video_fraction=fraction, pattern="poisson")
                   for fraction in (0.2, 0.5, 0.8)]
        trace = make(windows=windows)
        self.assertTrue(validate_trace(trace))
        start = 0.0
        for index, expected in enumerate((4, 10, 16)):
            manifest = trace["manifest"]["windows"][index]
            rows = [row for row in trace["workload"] if row["request_id"].startswith(f"w{index}-")]
            self.assertEqual(sum(row["kind"] == "video" for row in rows), expected)
            self.assertEqual(manifest["start_ms"], start)
            self.assertGreater(manifest["duration_ms"], 0)
            self.assertTrue(all(manifest["start_ms"] <= row["arrival_ms"] < manifest["start_ms"] + manifest["duration_ms"]
                                for row in rows))
            start += manifest["duration_ms"]
        self.assertEqual(start, trace["manifest"]["total_duration_ms"])

    def test_bursty_has_active_windows_idle_gaps_and_full_window_load(self):
        trace = make(pattern="bursty")
        manifest = trace["manifest"]
        active = manifest["windows"][0]["active_windows"]
        self.assertEqual(len(active), 2)
        first, second = active
        self.assertGreater(second["start_ms"], first["start_ms"] + first["duration_ms"])
        idle_start = first["start_ms"] + first["duration_ms"]
        self.assertTrue(all(not idle_start <= row["arrival_ms"] < second["start_ms"]
                            for row in trace["workload"]))
        self.assertEqual(len(trace["workload"]), 20)
        self.assertAlmostEqual(manifest["normalized_gpu_work"], 0.25)
        self.assertAlmostEqual(manifest["windows"][0]["duration_ms"],
                               2 * (second["start_ms"] - first["start_ms"]))

    def test_offline_and_bounded_quality_outputs(self):
        trace = make(pattern="offline_all", quality=2)
        self.assertTrue(all(row["arrival_ms"] == 0 for row in trace["workload"]))
        self.assertIsNone(trace["manifest"]["rate_rps"])
        self.assertAlmostEqual(trace["manifest"]["normalized_gpu_work"], 1.0)
        for kind in ("image", "video"):
            rows = [row for row in trace["workload"] if row["kind"] == kind]
            self.assertEqual(sum(row["output_mode"] == "file" for row in rows), 2)
            self.assertTrue(all(row["output_mode"] in ("file", "check") for row in rows))
        self.assertEqual(trace["manifest"]["quality_requests"], {"image": 2, "video": 2})

    def test_frozen_roundtrip_and_tamper_rejection(self):
        trace = make()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            write_trace(trace, path)
            write_trace(trace, path)
            self.assertEqual(load_trace(path), trace)
            changed = copy.deepcopy(trace)
            changed["workload"][0]["spec"]["prompt"] = "tampered"
            with self.assertRaisesRegex(ValueError, "frozen hash"):
                validate_trace(changed)
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                write_trace(make(seed=202), path)
            self.assertEqual(json.loads(path.read_text()), trace)

    def test_invalid_ratio_and_prompt_capacity_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exact integer"):
            build_trace(classes(), {"image": pool("diffusiondb"), "video": pool("vidprom")},
                        [dict(count=3, video_fraction=0.5, pattern="poisson")], slots=8, seed=1)
        with self.assertRaisesRegex(ValueError, "not enough distinct prompts"):
            build_trace(classes(), {"image": pool("diffusiondb", 2), "video": pool("vidprom", 2)},
                        [dict(count=20, video_fraction=0.2, pattern="poisson")], slots=8, seed=1)
        with self.assertRaisesRegex(ValueError, r"ratio\(t\) requires online"):
            make(windows=[dict(count=20, video_fraction=0.2, pattern="offline_all"),
                          dict(count=20, video_fraction=0.5, pattern="poisson")])


if __name__ == "__main__":
    unittest.main()
