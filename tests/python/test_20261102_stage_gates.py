"""Formal eligibility is recomputed from source records, never job status."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import native_exact
import native_quality
import paper_results
import stage_gates
from test_native_exact import record


def write(root, name, value):
    raw = value.encode() if isinstance(value, str) else json.dumps(value).encode()
    (root / name).write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(root):
    build = {"commit": "a" * 40, "core_sha256": "c" * 64, "image_digest": "sha256:" + "d" * 64}
    python_sources = {"ditserve/runtime/worker.py": "b" * 64}
    protocol = {"schema": "ditserve.formal-protocol.v1", "id": "fixture",
                "p0_required_cases": [{"model": "wan21", "shape_id": "wan21-dev"}],
                "shapes": {"wan21-dev": {"model": "wan21", "kind": "video", "height": 480,
                    "width": 832, "frames": 33, "steps": 20, "guidance_scale": 5.0}},
                "p0_diagnostics": {name: {"count": 1, "video_fraction": 1.0, "load": .1,
                    "pattern": "poisson", "video_slo_scale": 1.0, "image_slo_scale": 1.0,
                    "lazy_preemption": name == "no_preemption", "require_observed_stops": name == "contention"}
                    for name in ("contention", "no_preemption")}}
    smoke = {"schema": "ditserve.runtime-smoke.v1", "status": "passed", **build,
             "test_revision": build["commit"], "devices": 1, "sum": 524800, "matmul_value": 32,
             "python_sources": python_sources}
    job = {"name": "contract-fixture", "image": "registry/ditserve@" + build["image_digest"],
           "env": [{"name": "DITSERVE_BUILD", "value": build["commit"]}]}
    job_ref = write(root, "job.json", job)
    smoke.update(run_id=job["name"], execution_image=job["image"],
                 image_identity_source="submitted_job_spec_digest_pin", image_job_spec_sha256=job_ref["sha256"])
    manifest = {"schema": "ditserve.stage-evidence.v1", "build": build,
                "protocol": write(root, "protocol.json", protocol),
                "contract": {"runtime_smoke": write(root, "smoke.json", smoke),
                             "job_spec": job_ref,
                             "log": write(root, "contract.log", "Ran 120 tests in 1.0s\n\nOK\nRan 40 tests in 1.0s\n\nOK\n")},
                "runs": []}
    for role in ("preemption", "no_preemption"):
        run = record(stops=role == "preemption")
        benchmark = run["config"]["benchmark"]
        benchmark["preemption_diagnostic"] = "contention" if role == "preemption" else "no_preemption"
        controls = protocol["p0_diagnostics"][benchmark["preemption_diagnostic"]]
        benchmark.update({key: controls[key] for key in ("count", "video_fraction", "load", "pattern", "video_slo_scale", "image_slo_scale")})
        benchmark["p0_diagnostic_controls"] = controls
        benchmark["model_manifests"] = {"wan21": {"path": "/models/manifest.json", "sha256": "e" * 64}}
        run["source"]["files"] = python_sources
        run["config"]["workers"] = {"wan0": {"backend": "wan", "config": {
            "model_path": "/models/wan", "model_manifest_path": "/models/manifest.json", "model_manifest_sha256": "e" * 64}}}
        benchmark["formal_protocol"] = {"schema": protocol["schema"], "sha256": stage_gates.canonical_hash(protocol),
                                        "split": "development", "shape_ids": ["wan21-dev"], "role": role}
        for row in run["requests"]:
            row.update(arrival_ms=0.0, deadline_ms=1000.0, finish_ms=800.0, slo_ms=1000.0,
                       dataset_sample_id="sample0", **{"class": "wan21-dev"})
            row["outputs"][-1]["generation"].update(model_manifest_sha256="e" * 64, model_integrity="manifest_verified")
        trace = [{key: copy.deepcopy(run["requests"][0][key]) for key in (
            "request_id", "kind", "class", "arrival_ms", "deadline_ms", "slo_ms", "spec", "dataset_sample_id")}]
        trace[0].update(service_ms=1000.0, effective_slo_scale=1.0)
        digest = stage_gates.canonical_hash(trace)
        provenance = {"sha256": digest, "manifest": {"workload_sha256": digest, "windows": [controls]}}
        benchmark["trace"] = provenance
        run["config"]["phases"] = [{"id": "cluster", "options": {"forms": ["cfg2"],
            "lazy_preemption": controls["lazy_preemption"]}, "matrix": {"/policy": ["edf-cfg2", "block-cfg2"],
            "/workload": [trace], "/trace_provenance": [provenance]}, "profile": {
                "classes": {"wan21-dev": {"step_compute_ms": 45.0, "finish_ms": 100.0}}}}]
        manifest["runs"].append({"model": "wan21", "shape_id": "wan21-dev", "role": role,
                                  **write(root, role + ".json", run)})
    return manifest


class StageGateTests(unittest.TestCase):
    def test_p0_is_scoped_and_does_not_promote_unimplemented_stages(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = stage_gates.evaluate(fixture(root), root)
            self.assertEqual(result["stages"]["CONTRACT"]["status"], "passed")
            self.assertEqual(result["stages"]["P0"]["status"], "passed")
            self.assertEqual(result["stages"]["P0"]["allows"], ["p1_diagnostics"])
            self.assertTrue(all(result["stages"][f"P{i}"]["status"] == "pending" for i in range(1, 6)))

    def test_contract_precedes_p0_without_circular_dependency(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            value = fixture(root)
            value["runs"] = []
            result = stage_gates.evaluate(value, root)
            self.assertEqual(result["stages"]["CONTRACT"]["status"], "passed")
            self.assertIn("preemption", result["stages"]["CONTRACT"]["allows"])
            self.assertEqual(result["stages"]["P0"]["status"], "blocked")

    def test_unverified_weights_and_missing_shapes_cannot_pass(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            value = fixture(root)
            entry = value["runs"][0]
            run = json.loads((root / entry["path"]).read_text())
            for row in run["requests"]:
                row["outputs"][-1]["generation"]["model_integrity"] = "local_snapshot_not_rehashed"
            entry.update(write(root, entry["path"], run))
            result = stage_gates.evaluate(value, root)
            self.assertEqual(result["stages"]["P0"]["status"], "blocked")
            self.assertIn("weight", result["stages"]["P0"]["checks"][1]["reason"])

    def test_p0_checks_executed_controls_counts_weights_and_python_sources(self):
        def change(field, value):
            return lambda run: run["config"]["benchmark"].__setitem__(field, value)

        mutations = [change("load", .9), change("pattern", "bursty"), change("video_fraction", .5),
                     change("video_slo_scale", 2.5),
                     lambda run: run["config"]["phases"][0]["options"].__setitem__("lazy_preemption", True),
                     lambda run: run["requests"].append(copy.deepcopy(run["requests"][0])),
                     lambda run: [row.__setitem__("deadline_ms", 2000.0) for row in run["requests"]],
                     lambda run: [row["outputs"][-1]["generation"].__setitem__("model_manifest_sha256", "f" * 64)
                                  for row in run["requests"]],
                     lambda run: run["config"]["workers"]["wan0"]["config"].__setitem__("model_manifest_sha256", "f" * 64),
                     lambda run: run["source"]["files"].__setitem__("ditserve/runtime/worker.py", "f" * 64)]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index), TemporaryDirectory() as directory:
                root = Path(directory)
                value = fixture(root)
                entry = value["runs"][0]
                run = json.loads((root / entry["path"]).read_bytes())
                mutate(run)
                entry.update(write(root, entry["path"], run))
                self.assertEqual(stage_gates.evaluate(value, root)["stages"]["P0"]["status"], "blocked")

    def test_claiming_a_larger_frozen_count_cannot_qualify_a_single_pair(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            value = fixture(root)
            protocol = json.loads((root / "protocol.json").read_bytes())
            protocol["p0_diagnostics"]["contention"]["count"] = 40
            value["protocol"] = write(root, "protocol.json", protocol)
            for entry in value["runs"]:
                run = json.loads((root / entry["path"]).read_bytes())
                benchmark = run["config"]["benchmark"]
                benchmark["formal_protocol"]["sha256"] = stage_gates.canonical_hash(protocol)
                if entry["role"] == "preemption":
                    benchmark["count"] = benchmark["p0_diagnostic_controls"]["count"] = 40
                entry.update(write(root, entry["path"], run))
            result = stage_gates.evaluate(value, root)
            self.assertEqual(result["stages"]["P0"]["status"], "blocked")
            self.assertIn("workload count", result["stages"]["P0"]["checks"][1]["reason"])

    def test_generated_formal_configs_satisfy_the_executed_input_contract(self):
        from test_20261102_formal_benchmark import FormalBenchmarkTest
        FormalBenchmarkTest.setUpClass()
        helper = FormalBenchmarkTest()
        for diagnostic in ("contention", "no_preemption"):
            config = helper.build(split="development", seed=201, diagnostic=diagnostic)
            phase = config["phases"][0]
            run = {"config": config, "requests": [], "phases": [{"id": "cluster", "cells": []}]}
            for cell, policy in enumerate(phase["matrix"]["/policy"]):
                run["phases"][0]["cells"].append({"cell": cell, "policy": policy})
                for wanted in phase["matrix"]["/workload"][0]:
                    row = copy.deepcopy(wanted)
                    worker = config["workers"]["wan0" if row["kind"] == "video" else "image0"]
                    row.update(cell=cell, phase="cluster", outputs=[{"generation": {
                        "model_manifest_sha256": "a" * 64, "model_integrity": "manifest_verified",
                        "model_path": worker["config"]["model_path"]}}])
                    run["requests"].append(row)
            stage_gates.p0_controls(run, {"model": "wan21", "role": "preemption" if diagnostic == "contention" else diagnostic}, helper.p)

    def test_contract_partial_recheck_skips_wrong_image_and_suite_failure_are_blocked(self):
        for log in ("Ran 10 tests in 1s\n\nOK\n", "Ran 10 tests in 1s\n\nOK (skipped=1)\n",
                    "Ran 10 tests in 1s\n\nFAILED (errors=1)\n"):
            with TemporaryDirectory() as directory:
                root = Path(directory)
                value = fixture(root)
                value["contract"]["log"] = write(root, "contract.log", log)
                self.assertEqual(stage_gates.evaluate(value, root)["stages"]["CONTRACT"]["status"], "blocked")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            value = fixture(root)
            smoke = json.loads((root / "smoke.json").read_text())
            smoke["image_digest"] = "sha256:" + "f" * 64
            value["contract"]["runtime_smoke"] = write(root, "smoke.json", smoke)
            self.assertEqual(stage_gates.evaluate(value, root)["stages"]["CONTRACT"]["status"], "blocked")

    def test_receipt_revalidates_raw_evidence_and_action(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            value = fixture(root)
            result = stage_gates.evaluate(value, root)
            result["source_manifest"] = write(root, "manifest.json", value)
            write(root, "receipt.json", result)
            path = root / "receipt.json"
            stage_gates.verify_receipt(path, build_commit="a" * 7, required_stage="P0", action="p1_diagnostics")
            with self.assertRaisesRegex(ValueError, "authorize"):
                stage_gates.verify_receipt(path, required_stage="P0", action="main")
            with self.assertRaisesRegex(ValueError, "has not passed"):
                stage_gates.verify_receipt(path, required_stage="P3")
            (root / "preemption.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "recomputed"):
                stage_gates.verify_receipt(path)

    def test_exact_rejects_invalid_stop_events_and_changed_deadline(self):
        for duration in (-1, float("nan"), float("inf"), True):
            value = record()
            value["requests"][1]["preemption_stop_samples"][0]["stop_ms"] = duration
            with self.assertRaisesRegex(ValueError, "invalid preemption"):
                native_exact.compare(value)
        value = record()
        value["requests"][0]["preemption_stop_samples"] = [{"stop_ms": 1}]
        with self.assertRaisesRegex(ValueError, "reference unexpectedly"):
            native_exact.compare(value)
        value = record()
        value["requests"][1]["deadline_ms"] = 300
        with self.assertRaisesRegex(ValueError, "trace identity"):
            native_exact.compare(value)

    def test_quality_full_fixed_controls_match_adaptive_subsets(self):
        candidate = {"status": "completed", "source": {"reference_run_sha256": "r"}, "pairs": [
            {"dataset_sample_id": str(i), "seed": 301, "prompt": "p", "shape": [33, 480, 832, 20],
             "candidate_cfg_folded_steps": 2 if i < 2 else 3, "psnr_db": 31.0,
             "ssim": .95, "lpips_mean": .1} for i in range(4)]}
        controls = {}
        for count in (2, 3):
            control = copy.deepcopy(candidate)
            for row in control["pairs"]:
                row["candidate_cfg_folded_steps"] = count
            controls[count] = control
        result = native_quality.matched_fold_bootstrap(candidate, controls, resamples=100)
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["matched_pairs"], 4)
        controls[2]["pairs"] = controls[2]["pairs"][1:]
        with self.assertRaisesRegex(ValueError, "identities"):
            native_quality.matched_fold_bootstrap(candidate, controls, resamples=100)

    def test_paper_failure_rows_do_not_crash_or_hide_missing_latency(self):
        from test_20261102_paper_results import record as paper_record
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = paper_record()
            for row in run["requests"]:
                row["status"] = "failed"
            write(root, "run.json", run)
            row = paper_results.paper_row(root, {"run": "run.json", "label": "Failure", "policy": "edf-cfg2"})
            self.assertEqual(row["metrics"]["by_kind"]["video"]["ontime"], 0)
            self.assertIn("--- / ---", row["tex"])


if __name__ == "__main__":
    unittest.main()
