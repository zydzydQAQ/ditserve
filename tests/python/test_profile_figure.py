"""Evidence-selection contracts; synthetic scalars only, never GPU sampling."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation.wan_profile.figure import build_analysis, generate_figure


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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def independent_hardware():
    output = dict(sha256="a" * 64, finite=True)
    config = {"experiment": "wan-profile-corrected", "workers": {"model": {"backend": "wan_profile"}}}
    source = {"core_sha256": "b" * 64, "files": {"ditserve/models/wan/profile.py": "c" * 64}}
    clean = dict(mode="clean", status="ok", spec={"frames": 17}, output=dict(output))
    rank = dict(mode="hardware", status="ok", spec={"frames": 17}, output=dict(output),
                numerical_validation=dict(passed=True, reference_sha256="a" * 64))
    phase = dict(id="hardware_metrics", status="completed", qualification={"qualified": True},
                 final_occupancy=dict(status="exclusive_observed", foreign=[], owned=[]),
                 cells=[dict(operations=[dict(ranks=[clean]), dict(ranks=[rank])])])
    study = dict(schema="ditserve.run.v1", run_id="hardware02", experiment="wan-profile-corrected",
                 status="incomplete", config=config, config_sha256=digest(config), source=source, phases=[phase])
    wrapper = dict(schema="ditserve.wan_hardware_capture.v1", status="sampled", measured=True,
        hardware_phase_qualified=True, preflight=dict(ready=True, run_id=study["run_id"], experiment=study["experiment"],
            config_sha256=study["config_sha256"], source_sha256=digest(source),
            work_dir="/tmp/wan-profile-corrected/hardware02"),
        metrics={"ranges": [dict(frames=17, scope="request", valid=True,
            metrics=[dict(id="sm_active_pct", measured=True, status="sampled", time_weighted_mean_pct=42.,
                          sample_count=100, coverage_fraction=1.)])]})
    return study, wrapper


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

    def test_independent_hardware_uses_actual_run_with_incomplete_parent_status(self):
        run, old = legacy()
        hardware_study, wrapper = independent_hardware()
        software = new_trace()
        result = build_analysis(run, old, study=software, hardware=wrapper, hardware_study={"run": hardware_study})
        self.assertEqual(get(result, "activity", "request")["value"], 65.)
        row = get(result, "sm", "request/sm_active_pct")
        self.assertEqual(row["value"], 42.)
        self.assertEqual(row["validation_source"], "hardware_study")
        self.assertEqual(row["validation_source_pointers"], ["/run/phases/0/cells/0/operations/1/ranks/0"])
        self.assertEqual(result["hardware_study_status"], "incomplete")
        self.assertEqual(result["hardware_binding"]["status"], "verified")
        self.assertEqual(result["hardware_binding"]["identity"]["run_id"], "hardware02")
        self.assertEqual(result["hardware_binding"]["identity"]["source_sha256"], digest(hardware_study["source"]))
        self.assertIn("hardware_study", {source["id"] for source in result["sources"]})

    def test_explicit_missing_or_failed_hardware_study_never_falls_back(self):
        run, old = legacy()
        valid_study, wrapper = independent_hardware()
        software = new_trace()
        software["phases"].append(valid_study["phases"][0])  # Old validation would otherwise succeed.
        with tempfile.TemporaryDirectory() as directory:
            candidates = [None, {}, Path(directory) / "missing.json", dict(valid_study, status="failed")]
            for candidate in candidates:
                with self.subTest(candidate=str(candidate)[:80]):
                    result = build_analysis(run, old, study=software, hardware=wrapper, hardware_study=candidate)
                    self.assertIsNone(get(result, "sm", "request/sm_active_pct")["value"])
                    self.assertEqual(get(result, "sm", "request/sm_active_pct")["validation_source"], "hardware_study")
                    self.assertEqual(get(result, "sm", "request/sm_active_pct")["validation_source_pointers"], [])
                    self.assertEqual(get(result, "activity", "request")["value"], 65.)
                    self.assertEqual(result["hardware_binding"]["status"], "unverified")

    def test_independent_hardware_checks_phase_numeric_and_actual_gpu_occupancy(self):
        run, old = legacy()
        mutations = [
            lambda ph: ph.update(status="failed"),
            lambda ph: ph["qualification"].update(qualified=False),
            lambda ph: ph["final_occupancy"].update(status="unknown"),
            lambda ph: ph["final_occupancy"].update(foreign=[{"pid": 9}]),
            lambda ph: ph["final_occupancy"].update(owned=[{"pid": 8}]),
            lambda ph: ph["cells"][0]["operations"][1]["ranks"][0]["numerical_validation"].update(passed=False),
            lambda ph: ph["cells"][0]["operations"][1]["ranks"][0]["numerical_validation"].update(reference_sha256="d" * 64),
            lambda ph: ph["cells"][0]["operations"][0]["ranks"][0]["output"].update(sha256="e" * 64),
            lambda ph: ph["cells"][0]["operations"][1]["ranks"][0]["output"].update(finite=False),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, wrapper = independent_hardware()
                mutate(study["phases"][0])
                result = build_analysis(run, old, study=new_trace(), hardware=wrapper, hardware_study=study)
                self.assertIsNone(get(result, "sm", "request/sm_active_pct")["value"])

    def test_independent_hardware_rejects_mismatched_capture_source_config_and_run_identity(self):
        run, old = legacy()
        mutations = [
            lambda study, capture: capture["preflight"].update(config_sha256="d" * 64),
            lambda study, capture: capture["preflight"].update(source_sha256="d" * 64),
            lambda study, capture: capture["preflight"].update(run_id="run01"),
            lambda study, capture: capture["preflight"].update(work_dir="/tmp/wan-profile-corrected/run01"),
            lambda study, capture: capture.update(source_sha256="d" * 64),
            lambda study, capture: study["config"].update(changed=True),
            lambda study, capture: study["source"].update(changed=True),
            lambda study, capture: capture.pop("preflight"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                study, wrapper = independent_hardware()
                mutate(study, wrapper)
                result = build_analysis(run, old, study=new_trace(), hardware=wrapper, hardware_study=study)
                self.assertIsNone(get(result, "sm", "request/sm_active_pct")["value"])
                self.assertEqual(result["hardware_binding"]["status"], "unverified")
                self.assertTrue(result["hardware_binding"]["reason_codes"])

    def test_explicit_run_file_requires_actual_capture_work_directory(self):
        run, old = legacy()
        study, wrapper = independent_hardware()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / study["experiment"] / study["run_id"]
            work.mkdir(parents=True)
            path = work / "run.json"
            path.write_text(json.dumps(study))
            mismatched = build_analysis(run, old, hardware=wrapper, hardware_study=path)
            self.assertIsNone(get(mismatched, "sm", "request/sm_active_pct")["value"])
            wrapper["preflight"]["work_dir"] = str(work)
            matched = build_analysis(run, old, hardware=wrapper, hardware_study=path)
            self.assertEqual(get(matched, "sm", "request/sm_active_pct")["value"], 42.)
            source = next(row for row in matched["sources"] if row["id"] == "hardware_study")
            self.assertEqual(source["path"], str(path))

    def test_generate_figure_forwards_independent_hardware_validation(self):
        run, old = legacy()
        study, wrapper = independent_hardware()
        with patch("ditserve.evaluation.wan_profile.figure.render_figure", return_value={}) as render:
            result = generate_figure(run, old, "unused", study=new_trace(), hardware=wrapper, hardware_study=study)
            self.assertEqual(get(result, "sm", "request/sm_active_pct")["value"], 42.)
            self.assertEqual(render.call_args.args[0]["hardware_binding"]["identity"]["run_id"], "hardware02")


if __name__ == "__main__":
    unittest.main()
