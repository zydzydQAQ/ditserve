"""CPU-only accounting fixtures. All times/outputs here are synthetic test data.

These fixtures are never emitted as experiment results and do not run a model,
HTTP service, scheduler, GPU, or performance benchmark.
"""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import external_comparison as common


class ExternalComparisonTests(unittest.TestCase):
    def fixture(self, count=8, video="wan21"):
        base = Path(__file__).resolve().parents[2] / "configs/benchmarks"
        protocol = json.loads((base/"formal-mixed-v1.json").read_bytes())
        contract = json.loads((base/"external-reproduction-v1.json").read_bytes())
        models = {model: {"path": "/shared/data/DITSERVE/models/formal-manifests/"+model+".json",
                           "sha256": str(index+1)*64} for index, model in enumerate(("wan21", "wan22", "sd35"))}
        shapes = (["wan21-480p", "wan21-720p"] if video == "wan21" else ["wan22-small", "wan22-720p"]) + ["sd35-512", "sd35-1024"]
        rows = []
        for i in range(count):
            sid = shapes[i % len(shapes)]
            shape = protocol["shapes"][sid]
            spec = {k: shape[k] for k in ("height", "width", "frames", "steps", "guidance_scale")}
            if shape["kind"] == "video":
                spec.update(cache=False, threshold=0.0, max_continuous=1, negative_prompt="")
            else:
                spec.update(batch_size=1, max_sequence_length=256, negative_prompt="blurry, low quality")
            spec.update(prompt="  café\n 雨天 🐕  " + str(i), seed=123+i, request_id="r"+str(i))
            rows.append(dict(request_id="r"+str(i), arrival_ms=i*10.0, deadline_ms=i*10.0+100.0,
                slo_ms=100.0, service_ms=40.0, kind=shape["kind"], **{"class": sid}, spec=spec,
                trace_seed=301, dataset="synthetic-unit-fixture", dataset_sample_id="sample"+str(i),
                extra_frozen_provenance={"preserve": ["all", "fields"]}))
        sha = common.canonical(rows, ascii=True)
        manifest = dict(schema="ditserve.benchmark.mixed-workload.v1", workload_sha256=sha,
            seed=301, slots=8, counts=dict(common.Counter(r["kind"] for r in rows)),
            class_counts=dict(common.Counter(r["class"] for r in rows)), total_duration_ms=count*10.0,
            sources={"image": {"revision": "frozen", "sha256": "c"*64}, "video": {"revision": "frozen", "sha256": "d"*64}},
            windows=[{"synthetic_unit_fixture": True, "pattern": "poisson", "load": 0.6, "video_fraction": 0.5}], optional_future_metadata={"keep": True})
        provenance = {"path": "generated:frozen-formal-split", "sha256": sha, "manifest": manifest}
        study = {"experiment": "synthetic-unit-fixture-only", "workers": {},
            "phases": [{"id": "serve", "matrix": {"/policy": ["baseline", "candidate"],
                "/workload": [rows], "/trace_provenance": [provenance]}}],
            "benchmark": {"trace": copy.deepcopy(provenance), "model_manifests": copy.deepcopy(models),
                "formal_protocol": {"schema": protocol["schema"], "id": protocol["id"],
                    "sha256": common.canonical(protocol), "split": "test", "role": "serving",
                    "shape_ids": shapes, "profile_bundle_sha256": "e"*64,
                    "dataset_split_sha256": {"image": "c"*64, "video": "d"*64}}}}
        values = dict(study=study, protocol=protocol, contract=contract, models=models)
        bindings = {k: {"path": k+".json", "raw_sha256": common.file_sha(json.dumps(v).encode())} for k,v in values.items()}
        return values, bindings

    def exported(self, count=8, video="wan21"):
        values, bindings = self.fixture(count, video)
        return common.export_study(**values, bindings=bindings)

    def results(self, exported):
        return {"schema": common.RESULT_SCHEMA, "workload_sha256": exported["sha256"],
            "completion_boundary": common.BOUNDARY,
            "client_clock": {"id": "unit-clock", "origin_monotonic_ns": 10**12, "time_reference": "client_origin_ms"},
            "observed_until_ms": exported["clock_contract"]["measurement_horizon_ms"] + 100,
            "system": {"id": "synthetic-test-only", "source_manifest_sha256": "f"*64}, "requests": []}

    def completed(self, request, ready=None):
        spec = request["spec"]
        return {"request_id": request["request_id"], "source_request_sha256": request["source_request_sha256"],
            "clock_id": "unit-clock", "time_reference": "client_origin_ms", "status": "completed",
            "executed_spec_sha256": request["spec_sha256"], "model_manifest_sha256": request["model_manifest"]["sha256"],
            "dispatch_ms": request["arrival_ms"] + 3, "output_ready_ms": ready if ready is not None else request["arrival_ms"]+20,
            "output": {"decoded": True, "finite": True, "count": 1, "sha256": "a"*64,
                       **{k:spec[k] for k in ("height", "width", "frames")}}}

    def test_export_preserves_complete_trace_and_unicode_prompt_without_resampling(self):
        values, bindings = self.fixture()
        original = copy.deepcopy(values)
        result = common.export_study(**values, bindings=bindings)
        self.assertEqual(values, original)
        self.assertEqual(result["workload"], values["study"]["phases"][0]["matrix"]["/workload"][0])
        self.assertEqual(result["trace_provenance"], values["study"]["benchmark"]["trace"])
        for row, request in zip(result["workload"], result["requests"]):
            self.assertEqual(request["spec"], row["spec"])
            self.assertEqual(request["arrival_ms"], row["arrival_ms"])
            self.assertEqual(request["deadline_ms"], row["deadline_ms"])
            self.assertEqual(request["prompt_utf8_sha256"], common.file_sha(row["spec"]["prompt"].encode()))
        self.assertNotEqual(common.canonical(result["workload"]), common.canonical(result["workload"], ascii=True))
        self.assertFalse(result["execution_performed"])
        self.assertFalse(result["qualification_eligible"])
        self.assertTrue(result["batching"]["legacy_image_batch_limit_is_not_an_external_scheduler_limit"])
        common.verify_export(result)

    def test_pairwise_cells_and_formal_coverage_are_explicit(self):
        for video in ("wan21", "wan22"):
            result = self.exported(200, video)
            self.assertEqual(result["comparison_cell"]["id"], video+"-sd35")
            self.assertEqual(set(r["model"] for r in result["requests"]), {video, "sd35"})
            self.assertTrue(result["declared_formal_trace_candidate"])
            self.assertFalse(result["qualification_eligible"])
        self.assertFalse(self.exported(8)["declared_formal_trace_candidate"])
        values, bindings = self.fixture()
        values["study"]["benchmark"]["formal_protocol"]["shape_ids"].append("wan22-720p")
        with self.assertRaisesRegex(ValueError, "pairwise"):
            common.export_study(**values, bindings=bindings)

    def test_unfrozen_formal_load_is_not_a_declared_formal_candidate(self):
        values, bindings = self.fixture(200)
        for provenance in [values["study"]["benchmark"]["trace"],
                           values["study"]["phases"][0]["matrix"]["/trace_provenance"][0]]:
            provenance["manifest"]["windows"][0]["load"] = 99.0
        exported = common.export_study(**values, bindings=bindings)
        self.assertFalse(exported["declared_formal_trace_candidate"])
        self.assertNotIn("formal_trace_eligible", exported)
        self.assertIn("not independently audited", exported["formal_candidate_scope"])
        self.assertFalse(exported["qualification_eligible"])
        self.assertEqual(exported["offered_requests"], 200)

    def test_ambiguous_phases_multiple_workloads_and_matrix_mutation_rejected(self):
        values, bindings = self.fixture()
        second = copy.deepcopy(values["study"]["phases"][0]); second["id"] = "second"
        values["study"]["phases"].append(second)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            common.export_study(**values, bindings=bindings)
        result = common.export_study(**values, bindings=bindings, phase_id="second")
        self.assertEqual(result["selected_phase"], "second")
        second["matrix"]["/workload"].append(copy.deepcopy(second["matrix"]["/workload"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one workload"):
            common.export_study(**values, bindings=bindings, phase_id="second")
        second["matrix"]["/workload"].pop()
        second["matrix"]["/workload/0/spec/seed"] = [10, 20]
        with self.assertRaisesRegex(ValueError, "request-mutating"):
            common.export_study(**values, bindings=bindings, phase_id="second")

    def test_frozen_trace_protocol_models_and_changed_export_fail_closed(self):
        for target in ("trace", "protocol", "models"):
            values, bindings = self.fixture()
            if target == "trace":
                values["study"]["phases"][0]["matrix"]["/workload"][0][0]["spec"]["prompt"] += " changed"
            elif target == "protocol":
                values["protocol"]["shapes"]["wan21-480p"]["frames"] = 33
            else:
                values["models"]["wan21"]["sha256"] = "0"*64
            with self.assertRaises(ValueError):
                common.export_study(**values, bindings=bindings)
        value = self.exported()
        value["requests"][0]["deadline_ms"] += 1
        value = common.seal(value)
        with self.assertRaisesRegex(ValueError, "changed frozen"):
            common.verify_export(value)

    def test_all_offered_in_denominator_missing_failed_rejected_cancelled_timeout(self):
        exported = self.exported(8)
        result = self.results(exported)
        states = ["completed", "failed", "rejected", "cancelled", "timed_out", "not_executed", "incomplete"]
        for request, state in zip(exported["requests"], states):
            row = self.completed(request)
            row["status"] = state
            result["requests"].append(row)
        accounted = common.account_results(exported, result)
        self.assertEqual(accounted["overall"]["offered"], 8)
        self.assertEqual(accounted["overall"]["slo_met"], 1)
        self.assertEqual(accounted["overall"]["slo_attainment"], 1/8)
        self.assertEqual(accounted["overall"]["status_counts"]["missing"], 1)
        self.assertEqual(sum(x["offered"] for x in accounted["by_model"].values()), 8)
        self.assertEqual(sum(x["offered"] for x in accounted["by_shape"].values()), 8)
        self.assertEqual(accounted["status"], "incomplete")
        self.assertFalse(accounted["qualification_eligible"])

    def test_client_origin_counts_dispatch_lag_and_never_resets_deadlines(self):
        exported = self.exported(4); result = self.results(exported)
        request = exported["requests"][1]
        late = self.completed(request, request["deadline_ms"] + 1)
        late["dispatch_ms"] = request["deadline_ms"] - 1
        late["client_received_ms"] = late["output_ready_ms"] + 7
        result["requests"] = [late]
        accounted = common.account_results(exported, result)
        row = next(r for r in accounted["accounted_requests"] if r["request_id"] == request["request_id"])
        self.assertEqual(row["latency_ms"], 101)
        self.assertEqual(row["dispatch_lag_ms"], 99)
        self.assertEqual(row["wire_after_ready_ms"], 7)
        self.assertFalse(row["slo_met"])
        result["requests"][0]["output_ready_ms"] = request["deadline_ms"]
        self.assertEqual(common.account_results(exported, result)["overall"]["slo_met"], 1)

    def test_wrong_clock_duplicates_changed_parameters_and_undecoded_output_fail(self):
        exported = self.exported(4)
        mutations = [lambda r:r.update(clock_id="server-clock"), lambda r:r.update(time_reference="server_start_ms"),
            lambda r:r.update(deadline_ms=100000), lambda r:r.update(executed_spec_sha256="b"*64),
            lambda r:r.update(model_manifest_sha256="b"*64), lambda r:r["output"].update(decoded=False),
            lambda r:r["output"].update(frames=1), lambda r:r.update(output_ready_ms=10**12),
            lambda r:r.update(dispatch_ms=-1), lambda r:r["output"].update(count=True)]
        for mutation in mutations:
            result = self.results(exported); row = self.completed(exported["requests"][0]); mutation(row)
            result["requests"] = [row]
            accounted = common.account_results(exported, result)
            self.assertEqual(accounted["status"], "invalid_evidence")
            self.assertEqual(accounted["overall"]["offered"], 4)
            self.assertEqual(accounted["overall"]["slo_met"], 0)
        result = self.results(exported); row = self.completed(exported["requests"][0]); result["requests"] = [row, copy.deepcopy(row)]
        accounted = common.account_results(exported, result)
        self.assertEqual(accounted["overall"]["status_counts"]["invalid"], 1)
        self.assertEqual(accounted["overall"]["offered"], 4)

    def test_bad_envelope_and_unknown_results_cannot_hide_offered_requests(self):
        exported = self.exported(4)
        for change in ({"workload_sha256": "0"*64}, {"completion_boundary": "denoising_finished"},
                       {"client_clock": {}}, {"system": {}}, {"requests": {}}):
            result = self.results(exported); result.update(change)
            accounted = common.account_results(exported, result)
            self.assertEqual(accounted["status"], "invalid_evidence")
            self.assertEqual(accounted["overall"]["offered"], 4)
            self.assertEqual(accounted["overall"]["slo_met"], 0)
        result = self.results(exported); result["requests"] = [{"request_id": "unoffered"}]
        self.assertEqual(common.account_results(exported, result)["status"], "invalid_evidence")

    def test_common_horizon_does_not_extend_for_late_drain(self):
        exported = self.exported(4); result = self.results(exported)
        horizon = exported["clock_contract"]["measurement_horizon_ms"]
        result["requests"] = [self.completed(exported["requests"][0], horizon+10)]
        accounted = common.account_results(exported, result)
        self.assertEqual(accounted["overall"]["completed"], 1)
        self.assertEqual(accounted["overall"]["completed_in_window"], 0)
        self.assertEqual(accounted["overall"]["throughput_rps"], 0)
        self.assertEqual(accounted["overall"]["measurement_horizon_ms"], horizon)
        self.assertEqual(accounted["overall"]["latency_completed_only_ms"]["count"], 1)

    def test_cli_raw_bindings_and_no_overwrite(self):
        values, bindings = self.fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory); command = ["export"]
            for name, value in values.items():
                path = root/(name+".json"); path.write_text(json.dumps(value))
                command += ["--"+name, str(path), "--"+name+"-sha256", common.file_sha(path.read_bytes())]
            output = root/"export.json"; command += ["--output", str(output)]
            self.assertEqual(common.main(command), 0)
            common.verify_export(common.read_json(output.read_bytes()))
            with self.assertRaises(FileExistsError):
                common.main(command)
            with self.assertRaisesRegex(ValueError, "raw SHA256"):
                common.load_bound(output, "0"*64)
        for bad in ['{"a":1,"a":2}', '{"a":NaN}']:
            with self.assertRaises(ValueError):
                common.read_json(bad)


if __name__ == "__main__":
    unittest.main()
