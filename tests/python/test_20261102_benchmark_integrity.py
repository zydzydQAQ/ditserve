import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import native_exact
import native_metrics
import native_quality
import native_serving
import native_study
import compare_digests
from test_native_exact import record
from test_native_quality import run
from test_native_serving import pool, profile


class BenchmarkIntegrityTest(unittest.TestCase):
    def serving(self, **options):
        return native_serving.build_serving_config(
            "wan22", [f"GPU-{i}" for i in range(8)], options.pop("profile", profile()),
            {"image": pool("diffusiondb"), "video": pool("vidprom")},
            seed=201, load=1.0, pattern="poisson", video_fraction=0.5, count=20, **options)

    def test_failed_requests_cannot_earn_goodput_or_success_latency(self):
        rows = [dict(status="completed", arrival_ms=0, deadline_ms=100, finish_ms=50),
                dict(status="failed", arrival_ms=0, deadline_ms=100, finish_ms=1)]
        result = native_metrics.kind_summary(rows, 1000)
        self.assertEqual(result["ontime"], 1)
        self.assertEqual(result["deadline_attainment"], 0.5)
        self.assertEqual(result["goodput_rps"], 1)
        self.assertEqual(result["latency_ms"]["count"], 1)
        self.assertAlmostEqual(native_metrics.percentile([0, 10], .95), 9.5)
        self.assertEqual(native_metrics.percentile([0, 10], .5), 5)

    def test_order_randomization_does_not_change_frozen_trace(self):
        forward = self.serving()
        shuffled = self.serving(policy_order="random", order_seed=42)
        repeat = self.serving(policy_order="random", order_seed=42)
        reverse = self.serving(policy_order="reverse")
        self.assertEqual(shuffled["benchmark"]["policies"], repeat["benchmark"]["policies"])
        self.assertNotEqual(forward["benchmark"]["policies"], shuffled["benchmark"]["policies"])
        self.assertEqual(reverse["benchmark"]["policies"], list(reversed(forward["benchmark"]["policies"])))
        self.assertEqual(forward["benchmark"]["trace"], shuffled["benchmark"]["trace"])
        row = next(row for row in forward["phases"][0]["matrix"]["/workload"][0] if row["kind"] == "video")
        self.assertEqual(row["service_ms"], 20120.0)

    def test_atomic_batching_requires_measured_all_sizes_and_preserves_requests(self):
        with self.assertRaisesRegex(ValueError, "measured.*batch_ms"):
            self.serving(image_batch=2)
        measured = profile()
        measured["classes"]["sd35-preflight"]["batch_ms"] = {"2": 400, "3": 550}
        config = self.serving(profile=measured, image_batch=3)
        self.assertEqual(config["phases"][0]["options"]["image_batch"], 3)
        self.assertTrue(all(row["spec"]["batch_size"] == 1 for row in
                            config["phases"][0]["matrix"]["/workload"][0] if row["kind"] == "image"))
        with self.assertRaisesRegex(ValueError, "measured.*batch_ms"):
            self.serving(profile=measured, image_batch=4)
        calibration = native_study.build_calibration_config("wan22", [f"GPU-{i}" for i in range(4)],
                                                            image_batch_calibration=3)
        self.assertEqual([p["id"] for p in calibration["phases"]], ["cluster", "image-batch-2", "image-batch-3"])
        self.assertEqual(len(calibration["phases"][-1]["workload"]), 3)
        self.assertNotIn("static_cfg2sp3", calibration["phases"][0]["matrix"]["/policy"])

    def test_exact_rejects_input_execution_and_duplicate_identity_mismatch(self):
        for field, value in (("negative_prompt", "changed"), ("steps", 21), ("guidance_scale", 4.0)):
            changed = record()
            changed["requests"][1]["spec"][field] = value
            with self.assertRaisesRegex(ValueError, "identity|differs"):
                native_exact.compare(changed)
        changed = record()
        changed["requests"][1]["outputs"][-1]["generation"]["dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "identity"):
            native_exact.compare(changed)
        changed = record()
        changed["requests"].append(copy.deepcopy(changed["requests"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            native_exact.compare(changed)
        no_preemption = record(stops=False)
        no_preemption["config"]["benchmark"]["preemption_diagnostic"] = "no_preemption"
        self.assertEqual(native_exact.compare(no_preemption, require_preemption=False)["status"], "completed")
        unexpected_stops = copy.deepcopy(no_preemption)
        unexpected_stops["requests"][1]["preemption_stop_samples"] = [{"stop_ms": 11.0}]
        self.assertEqual(native_exact.compare(unexpected_stops, require_preemption=False)["status"], "incomplete")
        with self.assertRaisesRegex(ValueError, "controlled digest"):
            native_exact.compare(record(), require_preemption=False)
        old = record()
        for row in old["requests"]:
            row["outputs"][-1]["generation"].pop("runtime_identity")
        with self.assertRaisesRegex(ValueError, "runtime_identity"):
            native_exact.compare(old)
        legacy = native_exact.compare(old, allow_legacy_runtime=True)
        self.assertEqual(legacy["identity_qualification"], "unverified")
        self.assertFalse(legacy["weights_verified"])

    def test_quality_gates_identity_thresholds_and_infinite_psnr(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ref = np.zeros((1, 2, 4, 4, 3), dtype=np.float32)
            reference, candidate = run(root / "ref.npy", ref), run(root / "same.npy", ref)
            result = native_quality.compare_runs(reference, candidate, with_lpips=False, min_psnr=30, min_ssim=.9)
            self.assertEqual(result["psnr_threshold"]["passed"], 1)
            self.assertEqual(result["numeric_qualification"], "passed")
            bad = run(root / "bad.npy", np.ones_like(ref), fold=2)
            result = native_quality.compare_runs(reference, bad, with_lpips=False, min_psnr=30, min_ssim=.9)
            self.assertEqual(result["numeric_qualification"], "failed")
            bad["requests"][0]["outputs"][-1]["generation"]["revision"] = "different"
            result = native_quality.compare_runs(reference, bad, with_lpips=False)
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("identity", result["failures"][0]["error"])
            candidate["source"] = {"core_sha256": "b" * 64}
            with self.assertRaisesRegex(ValueError, "source identity"):
                native_quality.compare_runs(reference, candidate, with_lpips=False)

    def test_collective_comparison_cannot_ignore_changed_form_mismatches(self):
        reference = record(stops=False)
        reference.update(status="completed", source={"core_sha256": "a" * 64})
        candidate = copy.deepcopy(reference)
        second = copy.deepcopy(candidate["requests"][0])
        second["request_id"] = "v1"
        reference["requests"].append(copy.deepcopy(second))
        second["forms"] = ["sp1", "cfg2"]
        second["outputs"][-1]["output"]["array_sha256"] = "b" * 64
        candidate["requests"].append(second)
        result = compare_digests.compare(reference, candidate, ["edf-cfg2"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["policies"]["edf-cfg2"]["digest_mismatches_with_different_forms"], ["v1"])

    def test_comparison_cli_defaults_keep_aggregates_source_and_failures(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run.json"
            source.write_text(json.dumps(record()))
            for keep in (False, True):
                output = root / f"exact-{keep}.json"
                argv = ["native_exact.py", "--run", str(source), "--output", str(output)]
                with patch.object(sys, "argv", argv + (["--keep-pairs"] if keep else [])):
                    native_exact.main()
                summary = json.loads(output.read_text())
                self.assertEqual("pairs" in summary, keep)
                self.assertEqual(summary["matched_videos"], 1)
                self.assertEqual(summary["source"]["run_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            source.write_text(json.dumps(record(candidate_hash="b" * 64)))
            output = root / "mismatch.json"
            with patch.object(sys, "argv", ["native_exact.py", "--run", str(source), "--output", str(output)]):
                with self.assertRaises(SystemExit):
                    native_exact.main()
            self.assertEqual(json.loads(output.read_text())["mismatch_details"][0]["request_id"], "v0")

            reference = run(root / "ref.npy", np.zeros((1, 2, 4, 4, 3), dtype=np.float32))
            candidate = run(root / "cand.npy", np.ones((1, 2, 4, 4, 3), dtype=np.float32), fold=2)
            ref_path, cand_path = root / "ref.json", root / "cand.json"
            ref_path.write_text(json.dumps(reference))
            cand_path.write_text(json.dumps(candidate))
            output = root / "quality.json"
            with patch.object(sys, "argv", ["native_quality.py", "--reference-run", str(ref_path),
                    "--candidate-run", str(cand_path), "--output", str(output), "--no-lpips", "--min-psnr", "30"]):
                with self.assertRaises(SystemExit):
                    native_quality.main()
            summary = json.loads(output.read_text())
            self.assertNotIn("pairs", summary)
            self.assertNotIn("prompt", summary["threshold_failures"][0])
            self.assertEqual(summary["matched_pairs"], 1)
            self.assertIn("preemption_stop_ms", summary)
            self.assertEqual(summary["source"]["reference_run_sha256"], hashlib.sha256(ref_path.read_bytes()).hexdigest())

            digest_run = record(stops=False)
            digest_run.update(status="completed", source={"core_sha256": "a" * 64})
            ref_path.write_text(json.dumps(digest_run))
            cand_path.write_text(json.dumps(digest_run))
            output = root / "digest.json"
            with patch.object(sys, "argv", ["compare_digests.py", "--reference", str(ref_path),
                    "--candidate", str(cand_path), "--output", str(output), "--policies", "edf-cfg2"]):
                compare_digests.main()
            summary = json.loads(output.read_text())
            self.assertEqual(summary["source"]["reference_run_sha256"], hashlib.sha256(ref_path.read_bytes()).hexdigest())
            self.assertIn("preemption_stop_ms", summary["policies"]["edf-cfg2"])

    def test_cli_quality_arms_and_preemption_have_controlled_configuration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(json.dumps(profile()))
            for name in ("diffusiondb", "vidprom"):
                (root / f"{name}.json").write_text(json.dumps(pool(name)))
            common = ["native_serving.py", "--model", "wan22", "--profile", str(root / "profile.json"),
                      "--datasets", str(root), "--gpu-uuids", ",".join(f"GPU-{i}" for i in range(8)), "--count", "10"]
            configs = []
            for arm in ("exact", "fixed", "curvature"):
                dest = root / f"{arm}.json"
                with patch.object(sys, "argv", common + ["--quality-arm", arm, "--experiment", "controlled-quality", "--output", str(dest)]):
                    native_serving.main()
                configs.append(json.loads(dest.read_text()))
            self.assertEqual(len({c["benchmark"]["trace"]["sha256"] for c in configs}), 1)
            for config in configs:
                self.assertEqual(config["experiment"], "controlled-quality")
                self.assertEqual(config["phases"][0]["options"]["forms"], ["cfg2sp3"])
                self.assertFalse(config["phases"][0]["options"]["async_rank0_decode"])
            dest = root / "preempt.json"
            with patch.object(sys, "argv", common + ["--suite", "preempt", "--output", str(dest)]):
                native_serving.main()
            controlled = json.loads(dest.read_text())
            self.assertEqual(controlled["phases"][0]["options"]["forms"], ["cfg2"])
            self.assertEqual(controlled["benchmark"]["output_mode"], "digest")
            self.assertEqual(controlled["experiment"], "kairos-wan22-native-serving-s201-l1p0")

    def test_local_wrappers_keep_controlled_arms_and_evaluate_both_candidates(self):
        runtime_root = Path(os.environ.get("DITSERVE_KAIROS_RUNTIME_ROOT",
            Path(__file__).resolve().parents[3] / ".local/ditserve-kairos-rewrite-runtime"))
        scripts = runtime_root / "scripts"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "package/benchmarks").mkdir(parents=True)
            for name in ("kairos_native_approx.sh", "kairos_gfc_serving.sh"):
                (root / "scripts" / name).write_bytes((scripts / name).read_bytes())
            (root / "scripts/kairos_common.sh").write_text('''set -euo pipefail
export OUTPUT=$TEST_ROOT/output PACKAGE=$TEST_ROOT/package SCRIPTS=$TEST_ROOT/scripts
RUN_DIR=$OUTPUT/acceptance/$DITSERVE_RUN_ID
WAN_MODEL=${WAN_MODEL:-wan}
mkdir -p "$RUN_DIR"
''')
            (root / "scripts/kairos_native_serving.sh").write_text('''set -euo pipefail
printf '%s %s %s\\n' "${DITSERVE_QUALITY_ARM:-none}" "$DITSERVE_RUN_ID" "$DITSERVE_EXPERIMENT" >> "$TEST_ROOT/arms"
if [ "${SKIP_SOURCE:-0}" != 1 ]; then
  mkdir -p "$OUTPUT/.cache/results-work/$DITSERVE_EXPERIMENT/$DITSERVE_RUN_ID"
  printf '{}' > "$OUTPUT/.cache/results-work/$DITSERVE_EXPERIMENT/$DITSERVE_RUN_ID/run.json"
fi
''')
            (root / "package/benchmarks/native_quality.py").write_text('''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--reference-run', type=Path)
p.add_argument('--candidate-run', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--expected-fold-count', type=int)
p.add_argument('--max-lpips', type=float)
p.add_argument('--keep-pairs', action='store_true')
p.add_argument('--control-quality', action='append')
a, unused = p.parse_known_args()
assert a.reference_run.is_file() and a.candidate_run.is_file()
assert a.max_lpips == 0.1
a.output.write_text(json.dumps({'reference': str(a.reference_run), 'candidate': str(a.candidate_run),
    'expected_fold_count': a.expected_fold_count, 'keep_pairs': a.keep_pairs,
    'controls': a.control_quality or []}))
''')
            (root / "package/benchmarks/native_metrics.py").write_text('''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--run', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
assert a.run.is_file()
a.output.write_text(json.dumps({'run': str(a.run)}))
''')
            environment = {**os.environ, "TEST_ROOT": str(root), "DITSERVE_RUN_ID": "controlled",
                "DITSERVE_EXPERIMENT": "custom.experiment", "DITSERVE_CALIBRATE": "0", "DITSERVE_PROFILE": "fixture"}
            result = subprocess.run(["bash", str(root / "scripts/kairos_native_approx.sh")], env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([line.split()[0] for line in (root / "arms").read_text().splitlines()],
                             ["exact"] + ["fixed"] * 7 + ["curvature"] + ["none"] * 4)
            for count in range(8):
                name = f"fixed{count}"
                summary = json.loads((root / f"output/acceptance/controlled/quality-{name}.json").read_text())
                self.assertIn("custom.experiment/controlled-exact/run.json", summary["reference"])
                expected_run = "controlled-exact" if count == 0 else f"controlled-{name}"
                self.assertIn(f"custom.experiment/{expected_run}/run.json", summary["candidate"])
                self.assertEqual(summary["expected_fold_count"], count)
                self.assertTrue(summary["keep_pairs"])
            curvature = json.loads((root / "output/acceptance/controlled/quality-curvature.json").read_text())
            self.assertEqual(len(curvature["controls"]), 8)
            self.assertFalse(curvature["keep_pairs"])
            self.assertEqual(len(list((root / "output/acceptance/controlled").glob("metrics-*.json"))), 13)
            result = subprocess.run(["bash", str(root / "scripts/kairos_native_approx.sh")],
                env={**environment, "DITSERVE_RUN_ID": "missing", "SKIP_SOURCE": "1"}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("qualification is unverified", result.stderr)
            self.assertFalse(list((root / "output/acceptance/missing").glob("quality-*.json")))
            result = subprocess.run(["bash", str(root / "scripts/kairos_gfc_serving.sh")],
                env={**environment, "DITSERVE_POLICIES": "ditserve,edf-cfg2"}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires policy static-cfg2sp3", result.stderr)

    def test_generated_gfc_spec_includes_both_fixed_comparison_policies(self):
        runtime_root = Path(os.environ.get("DITSERVE_KAIROS_RUNTIME_ROOT",
            Path(__file__).resolve().parents[3] / ".local/ditserve-kairos-rewrite-runtime"))
        script = runtime_root / "tools/generate_batch_20261102.py"
        spec = importlib.util.spec_from_file_location("job_generator", script)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        with TemporaryDirectory() as directory:
            with patch.object(generator, "ROOT", Path(directory)), patch.object(generator, "resolve_sha", lambda ref: "0123456"), \
                    patch.object(sys, "argv", [str(script), "HEAD", "--stage", "gfc"]):
                generator.main()
            jobs = [json.loads(path.read_text()) for path in Path(directory).rglob("*.json")]
            serving = next(job for job in jobs if "gfc-serve" in job["name"])
            environment = {row["name"]: row["value"] for row in serving["env"]}
            self.assertEqual(set(environment["DITSERVE_POLICIES"].split(",")), {"static-cfg2sp3", "edf-cfg2"})


if __name__ == "__main__":
    unittest.main()
