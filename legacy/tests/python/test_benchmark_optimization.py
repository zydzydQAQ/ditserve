"""CPU-only evidence fixtures for paired GPU experiment planning and decisions.

These tests exercise provenance and decisions using small persisted arrays;
their constructed GPU identity labels are not performance measurements.
"""

import copy
from pathlib import Path
import tempfile
import unittest

from ditserve.benchmark.common import atomic_json, file_hash, read_json, seal
from ditserve.benchmark.evaluation import evaluate_run
from ditserve.benchmark.optimization import build_plan, execution_profile, summarize_runs, _stage_metrics

try:
    import numpy as np
except ImportError:
    np = None


ROOT = Path(__file__).resolve().parents[2]


def inputs():
    config = read_json(ROOT / "configs/benchmark/optimization.json")
    dataset = seal({"schema_version": "ditserve.benchmark.dataset.v1", "kind": "test-fixture",
        "samples": [{"sample_id": "dev", "prompt": "development", "service_split": "dev"}] +
                   [{"sample_id": "test-%d" % i, "prompt": "evaluation %d" % i, "service_split": "test"} for i in range(4)]})
    profiles = {}
    for name, attention, duration in (("math_serial", "MATH", 100.), ("flash_serial", "FLASH_ATTENTION", 20.)):
        identity = {"backend": "wan", "logical_device": "cuda:0", "gpu": {"uuid": "GPU-test-only", "name": "A100 test fixture",
            "total_memory_bytes": 40 * 1024 ** 3, "compute_capability": [8, 0]},
            "attention": {"processor": "native", "sdpa_backend": attention}, "model_id": config["model"]["id"],
            "revision": config["model"]["revision"], "precision": {"dit": "float16"}, "text_encoder_device": "cpu",
            "scheduler": "UniPCMultistepScheduler", "torch_version": "test", "cuda_runtime_version": "test", "driver_version": "test"}
        profiles[name] = seal({"schema_version": "ditserve.benchmark.profile.v1", "backend": "wan", "status": "completed",
            "measurement_kind": "measured_gpu", "execution_identity": identity, "config": copy.deepcopy(config),
            "dataset_sha256": dataset["sha256"], "dev_sample_ids": ["dev"],
            "shapes": {"short": dict(config["shapes"]["short"], service_ms=duration, prepare_ms=4., decode_ms=2.,
                step_ms=[(duration - 6) / 32] * 32, peak_increment_mb=4096., state_mb=512.)}})
    return config, dataset, profiles


def save_run(root, trace, duration=10., *, status="completed", observations=None, evaluate=True, quality_reference=None):
    """Persist complete synthetic arrays under a constructed measured-run schema."""
    root.mkdir()
    atomic_json(root / "trace.json", trace)
    rows, references = [], {}
    now = 0.
    for generation in trace["requests"]:
        rid = generation["request_id"]
        artifact = {"generation": copy.deepcopy(generation), "path_history": ["compute"] * generation["steps"]}
        for kind, array in (("latent", np.ones((1, 2), dtype=np.float32)),
                            ("output", np.full((1, 1, 2, 2, 3), .5, dtype=np.float32))):
            path = root / (rid + "-" + kind + ".npy")
            np.save(path, array, allow_pickle=False)
            artifact[kind + "_path"] = str(path)
            artifact[kind + "_sha256"] = file_hash(path)
        references[rid] = artifact
        now = max(now, generation["arrival_ms"])
        start = now
        events = [{"kind": "prepare", "start_ms": now, "end_ms": now + duration * .2,
                   "ms": duration * .2, "cpu_prompt_encode_ms": duration * .18}]
        now += duration * .2
        for index in range(generation["steps"]):
            step_ms = duration * .7 / generation["steps"]
            events.append({"kind": "step", "step_index": index, "start_ms": now, "end_ms": now + step_ms, "ms": step_ms})
            now += step_ms
        events.append({"kind": "decode", "start_ms": now, "end_ms": now + duration * .1, "ms": duration * .1})
        now += duration * .1
        rows.append({"request_id": rid, "shape_id": generation["shape_id"], "generation": copy.deepcopy(generation),
            "arrival_ms": generation["arrival_ms"], "deadline_ms": generation["deadline_ms"],
            "submit_ms": generation["arrival_ms"], "start_ms": start, "finish_ms": now,
            "status": "completed", "events": events, "artifact": artifact})
    policy = {"status": status, "horizon_ms": now, "requests": rows}
    if observations is not None:
        policy["observations"] = observations
    run = {"schema_version": "ditserve.benchmark.run.v1", "backend": "wan", "measured": True,
        "measurement_kind": "measured_gpu", "status": status, "purpose": "online_serving", "serving_performance_evidence": True,
        "trace_sha256": trace["sha256"], "execution_identity": trace["execution_profile"]["execution_identity"],
        "execution_options": trace["optimization"]["arguments"], "execution_profile_sha256": trace["execution_profile_sha256"],
        "references": references, "policies": {"fcfs": policy}}
    if quality_reference is not None:
        baseline = read_json(quality_reference)
        run["backend_quality_gate"] = seal({"schema_version": "ditserve.benchmark.backend_quality.v1", "status": "passed",
            "reference_run_path": str(quality_reference), "reference_run_sha256": baseline["sha256"],
            "reference_run_file_sha256": file_hash(quality_reference), "epsilon": .05,
            "requests": {rid: {"status": "passed", "candidate_sha256": artifact["output_sha256"],
                "reference_sha256": baseline["references"][rid]["output_sha256"],
                "metrics": {"status": "ok", "lpips_status": "ok", "finite": True, "range_valid": True, "lpips_mean": 0.}}
                for rid, artifact in references.items()}})
    run = seal(run)
    atomic_json(root / "run.json", run)
    if evaluate:
        evaluate_run(root)
    return root / "run.json"


class OptimizationPlanTests(unittest.TestCase):
    def test_pilot_freezes_one_shared_workload_and_distinct_cost_profiles(self):
        config, dataset, profiles = inputs()
        plan = build_plan(config, dataset, profiles, remaining_gpu_seconds=4853.)
        self.assertFalse(plan["executes_gpu_work"])
        self.assertEqual(len(plan["groups"]), 1)
        self.assertEqual(len(plan["deferred_groups"]), 11)
        group = plan["groups"][0]
        self.assertEqual(group["scenario"]["name"], "near")
        traces = group["traces"]
        for trace in traces.values():
            self.assertEqual(trace["requests"], traces["math_serial"]["requests"])
            self.assertEqual(trace["profile"]["sha256"], profiles["flash_serial"]["sha256"])
            self.assertEqual(len(trace["requests"]), 3)
        self.assertEqual(execution_profile(traces["math_serial"])["shapes"]["short"]["service_ms"], 100.)
        self.assertEqual(execution_profile(traces["flash_serial"])["shapes"]["short"]["service_ms"], 20.)
        self.assertEqual(traces["flash_prefetch"]["optimization"]["profile_alias_of"], "flash_serial")
        self.assertEqual(group["variants"]["flash_prefetch"]["arguments"]["text_prefetch"], 2)
        self.assertTrue(plan["budget"]["fits_estimate"])

    def test_expanded_is_three_seeds_and_four_conditions_but_never_runs(self):
        config, dataset, profiles = inputs()
        plan = build_plan(config, dataset, profiles, stage="expanded", remaining_gpu_seconds=10.)
        self.assertEqual(len(plan["groups"]), 12)
        self.assertEqual(plan["deferred_groups"], [])
        self.assertFalse(plan["budget"]["fits_estimate"])
        self.assertFalse(plan["executes_gpu_work"])
        self.assertEqual({g["seed"] for g in plan["groups"]}, set(config["trace_seeds"]))
        for group in plan["groups"]:
            self.assertEqual(len(group["traces"]), 3)
            self.assertEqual(len(group["traces"]["math_serial"]["requests"]), 6)

    def test_wrong_attention_gpu_dataset_or_incomplete_profile_rejected(self):
        for field in ("attention", "gpu", "dataset", "status", "synthetic"):
            config, dataset, profiles = inputs()
            profile = copy.deepcopy(profiles["math_serial"])
            if field == "attention":
                profile["execution_identity"]["attention"]["sdpa_backend"] = "FLASH_ATTENTION"
            elif field == "gpu":
                profile["execution_identity"]["gpu"]["uuid"] = "other-physical-gpu"
            elif field == "dataset":
                profile["dataset_sha256"] = "wrong"
            elif field == "status":
                profile["status"] = "partial"
            else:
                profile["measurement_kind"] = "synthetic_virtual_fixture"
            profiles["math_serial"] = seal(profile)
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_plan(config, dataset, profiles)

    def test_execution_profile_does_not_allow_hash_or_shared_request_tampering(self):
        config, dataset, profiles = inputs()
        trace = build_plan(config, dataset, profiles)["groups"][0]["traces"]["math_serial"]
        ordinary = {k: v for k, v in trace.items() if k not in ("execution_profile", "execution_profile_sha256", "optimization")}
        self.assertEqual(execution_profile(ordinary), ordinary["profile"])
        for field in ("hash", "request", "missing"):
            altered = copy.deepcopy(trace)
            if field == "hash":
                altered["execution_profile_sha256"] = "wrong"
            elif field == "request":
                altered["requests"][0]["seed"] += 1
            else:
                del altered["execution_profile"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                execution_profile(seal(altered))

    def test_empty_or_unobserved_stage_metrics_remain_unknown(self):
        metrics = _stage_metrics({"requests": [{"request_id": "r", "events": []}]})
        for name in ("cpu_prompt_encode_ms", "step_ms", "decode_ms", "snapshot_count", "restore_count", "step_preemptions", "max_gpu_ready"):
            self.assertIsNone(metrics[name], name)

    def test_text_prefetch_events_exclude_cleanup_and_do_not_double_count_prepare(self):
        result = _stage_metrics({"horizon_ms": 100., "requests": [{"request_id": "r", "events": [
            {"kind": "text_encode", "ms": 30.}, {"kind": "prepare", "ms": 2., "cpu_prompt_encode_ms": 0.},
            {"kind": "text_encode", "ms": 90., "outside_policy_horizon": True},
            {"kind": "decode", "ms": 80., "outside_policy_horizon": True}]}]})
        self.assertEqual(result["cpu_prompt_encode_ms"], 30.)
        self.assertEqual(result["cpu_work_fraction_of_horizon"], .3)
        self.assertIsNone(result["decode_ms"])


@unittest.skipIf(np is None, "numpy optional")
class OptimizationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config, self.dataset, self.profiles = inputs()
        self.traces = build_plan(self.config, self.dataset, self.profiles)["groups"][0]["traces"]

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, variant, **kwargs):
        if variant != "math_serial" and "quality_reference" not in kwargs:
            candidate = self.root / "math_serial" / "run.json"
            if candidate.is_file():
                kwargs["quality_reference"] = candidate
        return save_run(self.root / variant, self.traces[variant], **kwargs)

    def test_full_qualified_ablation_includes_pairs_but_does_not_invent_gates(self):
        paths = [self.save("math_serial", duration=30.), self.save("flash_serial", duration=10.),
                 self.save("flash_prefetch", duration=8.)]
        result = summarize_runs(paths)
        self.assertTrue(all(pair["status"] == "included" for pair in result["pairs"]))
        self.assertGreater(result["pairs"][0]["mean_latency_improvement_fraction"], 0.)
        self.assertEqual(result["gates"]["paired_ablation"]["status"], "measured")
        self.assertEqual(result["gates"]["batch_2"]["status"], "insufficient_evidence")
        self.assertEqual(result["gates"]["state_policy"]["status"], "insufficient_evidence")
        self.assertEqual(result["gates"]["quality_scope"]["status"], "cross_attention_gate_passed")
        for attempt in result["attempts"]:
            self.assertEqual(attempt["stages"]["snapshot_count"], 0)
            self.assertEqual(attempt["stages"]["step_preemptions"], 0)
            self.assertIsNone(attempt["stages"]["max_gpu_ready"])

    def test_missing_or_partial_arm_cannot_claim_a_complete_group(self):
        paths = [self.save("math_serial"), self.save("flash_serial", status="partial")]
        result = summarize_runs(paths)
        self.assertTrue(all(pair["status"] == "excluded" for pair in result["pairs"]))
        self.assertEqual(result["gates"]["paired_ablation"]["status"], "needs_measurement")
        self.assertIsNone(result["pairs"][0]["mean_latency_improvement_fraction"])

    def test_single_successful_pair_does_not_complete_three_arm_ablation(self):
        result = summarize_runs([self.save("math_serial"), self.save("flash_serial")])
        self.assertEqual(result["pairs"][0]["status"], "included")
        self.assertEqual(result["gates"]["paired_ablation"]["status"], "needs_measurement")

    def test_missing_quality_and_changed_artifacts_exclude_the_pair(self):
        left = self.save("math_serial")
        right = self.save("flash_serial", evaluate=False)
        result = summarize_runs([left, right])
        self.assertEqual(result["pairs"][0]["status"], "excluded")
        evaluate_run(right)
        run = read_json(right)
        video = Path(run["policies"]["fcfs"]["requests"][0]["artifact"]["output_path"])
        video.write_bytes(b"changed after evaluation")
        result = summarize_runs([left, right])
        self.assertEqual(result["pairs"][0]["status"], "excluded")
        self.assertIn("reference_quality_not_all_passed", result["attempts"][1]["eligibility_reasons"])

    def test_stale_evaluation_fails_and_duplicate_attempts_are_not_merged(self):
        path = self.save("math_serial")
        with self.assertRaisesRegex(ValueError, "multiple attempts"):
            summarize_runs([path, path])
        evaluation = read_json(path.parent / "evaluation.json")
        evaluation["run_sha256"] = "wrong"
        atomic_json(path.parent / "evaluation.json", seal(evaluation))
        with self.assertRaisesRegex(ValueError, "different run"):
            summarize_runs([path])

    def test_actual_observations_only_enable_bounded_feasibility_work(self):
        observations = {"step_preemptions": 2, "snapshot_count": 1, "restore_count": 1, "max_gpu_ready": 2}
        self.save("math_serial")
        result = summarize_runs([self.save("flash_prefetch", observations=observations)])
        self.assertEqual(result["gates"]["batch_2"]["status"], "profile_feasibility")
        self.assertFalse(result["gates"]["batch_2"]["batch_2_capacity_proven"])
        self.assertEqual(result["gates"]["state_policy"]["status"], "profile_switch_costs")
        self.assertFalse(result["gates"]["state_policy"]["cold_quality_proven"])

    def test_cross_attention_gate_and_actual_execution_arguments_are_required(self):
        left = self.save("math_serial")
        right = self.save("flash_serial", quality_reference=None)
        result = summarize_runs([left, right])
        self.assertIn("cross_attention_quality_not_passed", result["attempts"][1]["eligibility_reasons"])
        run = read_json(right)
        run["execution_options"]["text_prefetch"] = 2
        atomic_json(right, seal(run))
        evaluate_run(right)
        result = summarize_runs([left, right])
        self.assertIn("execution_arguments_mismatch", result["attempts"][1]["eligibility_reasons"])

    def test_changed_math_quality_reference_invalidates_flash_gate(self):
        left = self.save("math_serial")
        right = self.save("flash_serial")
        baseline = read_json(left)
        baseline["changed"] = True
        atomic_json(left, seal(baseline))
        result = summarize_runs([right])
        self.assertEqual(result["attempts"][0]["backend_quality"]["reason"], "backend_quality_reference_run_changed")

    def test_empty_summary_keeps_next_actions_unverified_and_output_immutable(self):
        output = self.root / "summary.json"
        result = summarize_runs([], output)
        self.assertEqual(result["gates"]["cpu_pipeline"]["status"], "unmeasured")
        self.assertEqual(result["gates"]["paired_ablation"]["included_pair_count"], 0)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            summarize_runs([], output)


if __name__ == "__main__":
    unittest.main()
