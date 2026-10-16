"""Evidence-selection contracts; synthetic scalars only, never GPU sampling."""

import json
from pathlib import Path
import tempfile
import unittest

from ditserve.evaluation.wan_profile.figure import build_analysis


def legacy():
    run = {"samples": [dict(mode="baseline", frames=17, e2e_ms=value) for value in (1000., 2000., 3000.)]}
    analysis = {"stages": [dict(frames=17, stage="text_encode", duration_ms=500., source_field="host_enqueue_ms")],
        "memory": [dict(frames=17, stage="transformer", peak_allocated_bytes=1024**3)],
        "summary": {"scaling": {"flash_ms": {"17": 250.}, "named_gemm_ms": {"17": 300.}}}}
    return run, analysis


def new_trace(*, timeline=True, attribution=True, numerical=True):
    rank = dict(mode="trace", spec={"frames": 17}, numerical_validation={"passed": numerical},
        instrumentation_validation={"passed": attribution}, profiler={
            "timeline_status": "measured" if timeline else "unverified",
            "attribution_status": "measured" if attribution else "unverified",
            "request": {"kernel_active_pct": 65.}, "gpu_window": {"kernel_active_pct": 95.},
            "transformer_kernel_duration_ms": 1000.,
            "transformer_kernel_categories": [dict(name=name, share_pct=value) for name, value in (
                ("self_attention", 50.), ("cross_attention", 10.), ("ffn", 30.), ("transformer_other", 10.))]})
    return {"phases": [{"status": "completed", "qualification": {"qualified": True},
                        "final_occupancy": {"status": "exclusive_observed", "foreign": []},
                        "cells": [{"operations": [{"ranks": [rank]}]}]}]}


def get(analysis, panel, series, frames=17):
    return next(row for row in analysis["metrics"] if row["panel"] == panel
                and row["series"] == series and row["frames"] == frames)


class ProfileFigureTests(unittest.TestCase):
    def test_preview_uses_history_and_keeps_new_metrics_null(self):
        run, old = legacy()
        # The previously incorrect values must never be considered for E or F.
        run["samples"].append(dict(mode="profiler", frames=17, kernel_active_pct=100., e2e_ms=9.))
        result = build_analysis(run, old, study={"status": "waiting_resource"})
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["study_status"], "waiting_resource")
        self.assertEqual(get(result, "e2e", "E2E")["value"], 2.)
        self.assertEqual(get(result, "e2e", "E2E")["minimum"], 1.)
        self.assertEqual(get(result, "e2e", "E2E")["maximum"], 3.)
        self.assertEqual(get(result, "e2e", "E2E")["n"], 3)
        self.assertEqual(get(result, "memory", "transformer")["value"], 1.)
        self.assertIsNone(get(result, "activity", "request")["value"])
        self.assertIsNone(get(result, "named_kernels", "Flash Attention", 33)["value"])
        self.assertFalse(result["historical_invalid_metrics_used"])
        self.assertEqual(get(result, "e2e", "E2E")["source_pointers"],
                         ["/samples/0/e2e_ms", "/samples/1/e2e_ms", "/samples/2/e2e_ms"])

    def test_new_trace_requires_numerical_and_independent_metric_validation(self):
        run, old = legacy()
        good = build_analysis(run, old, study=new_trace())
        self.assertEqual(get(good, "activity", "request")["value"], 65.)
        self.assertEqual(get(good, "shares", "self_attention")["value"], 50.)
        self.assertEqual(get(good, "shares", "other")["value"], 10.)
        partial = build_analysis(run, old, study=new_trace(attribution=False))
        self.assertEqual(get(partial, "activity", "request")["value"], 65.)
        self.assertIsNone(get(partial, "shares", "ffn")["value"])
        invalid = build_analysis(run, old, study=new_trace(numerical=False))
        self.assertIsNone(get(invalid, "activity", "request")["value"])
        self.assertIsNone(get(invalid, "shares", "self_attention")["value"])

    def test_archived_study_wrapper_and_live_json_are_supported(self):
        run, old = legacy()
        wrapped = build_analysis(run, old, study={"run": new_trace()})
        self.assertTrue(get(wrapped, "activity", "request")["source_pointers"][0].startswith("/run/phases/"))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run.json"
            source.write_text(json.dumps(new_trace()))
            result = build_analysis(run, old, study=directory)
            self.assertEqual(get(result, "activity", "request")["value"], 65.)
            provenance = next(row for row in result["sources"] if row["id"] == "study")
            self.assertEqual(provenance["bytes"], source.stat().st_size)
            self.assertEqual(len(provenance["sha256"]), 64)

    def test_unverified_hardware_estimate_is_not_promoted_to_measurement(self):
        run, old = legacy()
        hardware = {"ranges": [dict(frames=17, scope="request", valid=True, metrics=[
            dict(id="sm_active_pct", measured=True, status="sampled", time_weighted_mean_pct=0.,
                 sample_count=100, coverage_fraction=.99),
            dict(id="tensor_active_pct", measured=False, status="unknown", time_weighted_mean_pct=None,
                 covered_time_mean_estimate_pct=80.)])]}
        study = new_trace()
        study["phases"][0]["cells"][0]["operations"][0]["ranks"].append(
            dict(mode="hardware", status="ok", spec={"frames": 17}, numerical_validation={"passed": True}))
        result = build_analysis(run, old, hardware=hardware, study=study)
        self.assertEqual(get(result, "sm", "request/sm_active_pct")["value"], 0.)
        self.assertEqual(get(result, "sm", "request/sm_active_pct")["status"], "measured")
        self.assertIsNone(get(result, "tensor", "request/tensor_active_pct")["value"])
        self.assertIsNone(get(result, "sm", "denoise/sm_active_pct")["value"])
        missing_validation = build_analysis(run, old, hardware=hardware)
        self.assertIsNone(get(missing_validation, "sm", "request/sm_active_pct")["value"])
        study["phases"][0]["status"] = "failed"
        contaminated = build_analysis(run, old, hardware=hardware, study=study)
        self.assertIsNone(get(contaminated, "sm", "request/sm_active_pct")["value"])

    def test_inconsistent_share_total_is_not_drawn(self):
        run, old = legacy()
        study = new_trace()
        study["phases"][0]["cells"][0]["operations"][0]["ranks"][0]["profiler"]["transformer_kernel_categories"][0]["share_pct"] = 51.
        result = build_analysis(run, old, study=study)
        self.assertIsNone(get(result, "shares", "self_attention")["value"])

    def test_hardware_capture_wrapper_requires_top_level_success(self):
        run, old = legacy()
        study = new_trace()
        study["phases"][0]["cells"][0]["operations"][0]["ranks"].append(
            dict(mode="hardware", status="ok", spec={"frames": 17}, numerical_validation={"passed": True}))
        wrapper = dict(schema="ditserve.wan_hardware_capture.v1", status="sampled", measured=True,
                       hardware_phase_qualified=True, metrics={"ranges": [dict(frames=17, scope="request", valid=True,
                           metrics=[dict(id="sm_active_pct", measured=True, status="sampled", time_weighted_mean_pct=42.)])]})
        good = build_analysis(run, old, study=study, hardware=wrapper)
        self.assertEqual(get(good, "sm", "request/sm_active_pct")["value"], 42.)
        self.assertEqual(get(good, "sm", "request/sm_active_pct")["source_pointers"],
                         ["/metrics/ranges/0/metrics/0/time_weighted_mean_pct"])
        for changes in ({"status": "failed"}, {"hardware_phase_qualified": False},
                        {"measured": False}, {"failure": "capture failed after collection"}):
            with self.subTest(changes=changes):
                bad = build_analysis(run, old, study=study, hardware=dict(wrapper, **changes))
                self.assertIsNone(get(bad, "sm", "request/sm_active_pct")["value"])


if __name__ == "__main__":
    unittest.main()
