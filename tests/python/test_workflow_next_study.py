"""Run03 input, resource, resume and admission contracts; never start CUDA."""

import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation import workflow_next_report as report
from ditserve.evaluation.workflow_budget import atomic_json
from ditserve.runtime.client import CoreBridge

REPO = Path(__file__).resolve().parents[2]
UUIDS = [f"GPU-00000000-0000-0000-0000-{i:012}" for i in range(2)]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), REPO / "docker" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_script("run-next-study.py")
suite = load_script("run-workflow-suite.py")


class NextStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = dict(root=str(self.root), run_id="run03", source=str(REPO),
            source_digest="fixture", source_revisions=[], budget_seconds=28800,
            family_caps_seconds=api.CAPS, image="sha256:fixture", jobs=[], decisions=[], selections={},
            gpu_mapping=[dict(ordinal=i, uuid=UUIDS[i], name="RTX 4090") for i in range(2)])
        self.profile = api.read(REPO / "configs/profiles/rtx4090-sp1-local.json")
        self.capacities = dict(video=.08, mixed=.3)

    def generate(self):
        api.initial_jobs(self.study)
        api.measured_jobs(self.study, self.profile)
        self.study["selections"].update(screen_depth=dict(depth=2), cpu=dict(cores=16, depth=2))
        api.native_followups(self.study, self.profile, self.capacities)
        api.quantum_screen_job(self.study, self.profile, self.capacities)
        api.mixed_jobs(self.study, self.profile, self.capacities["mixed"], 300)
        api.mixed_jobs(self.study, self.profile, self.capacities["mixed"], 300, confirmation=True)
        api.ablation_jobs(self.study, self.profile, self.capacities["mixed"], 300)
        api.elastic_jobs(self.study, self.profile, self.capacities["mixed"])
        api.elastic_repeats(self.study, self.profile, self.capacities["mixed"])
        api.extension_jobs(self.study)

    def test_balanced_frozen_poisson_and_burst_trajectories(self):
        for pattern in ("poisson", "bursty"):
            rows = api.trajectory(self.profile, 3301, pattern=pattern)
            self.assertEqual(rows, api.trajectory(self.profile, 3301, pattern=pattern))
            self.assertEqual([r["arrival_ms"] for r in rows], sorted(r["arrival_ms"] for r in rows))
            self.assertEqual(sum(r["kind"] == "image" for r in rows), 32)
            self.assertEqual([sum(r["spec"].get("frames") == n for r in rows) for n in (17, 33)], [4, 4])
            self.assertTrue(all(r["deadline_ms"] > r["arrival_ms"] for r in rows))
            if pattern == "bursty":
                self.assertTrue(all(r["arrival_ms"] % 80000 < 16000 for r in rows))

    def test_every_generated_input_is_bound_and_single_gpu_uses_physical_one(self):
        self.generate()
        for job in self.study["jobs"]:
            self.assertEqual(api.digest(api.read(job["input"])), job["configuration_sha256"])
            self.assertEqual(suite.selected_devices(job), job["devices"])
            self.assertEqual([api.binding(Path(r["path"])) for r in job["input_bindings"]], job["input_bindings"])
            if job.get("command_type") == "native":
                config = api.read(job["input"])
                self.assertTrue(all(p["artifact_dependencies"][0]["closed"] is False for p in config["phases"]))
                if job["devices"] == [1]:
                    self.assertTrue(all(w["gpu"] == 0 for w in config["workers"].values()))
                    self.assertEqual(api.read(job["environment_input"])["gpus"]["uuids"], [UUIDS[1]])

    def test_cost_filter_keeps_complete_shapes_and_removes_81(self):
        api.initial_jobs(self.study)
        cost = api.read(report.current_job(self.study, "cost")["input"])
        actions = cost["phases"][0]["actions"]
        self.assertEqual(sum(a["op"] == "prepare" for a in actions), 2)
        self.assertEqual(sum(a["op"] == "memory" and a.get("reset_peak", False) for a in actions), 2)
        self.assertNotIn("81", " ".join(a.get("state_id", "") for a in actions))

    def test_screening_validates_all_six_arms_and_reverses_order(self):
        from ditserve.evaluation.workflow_serving_encoder import validate_config
        api.initial_jobs(self.study)
        arms = [api.read(report.current_job(self.study, f"cpu-screen-r{i}")["input"])["arms"] for i in (0, 1)]
        self.assertEqual([a["id"] for a in arms[0]], list(reversed([a["id"] for a in arms[1]])))
        for group in arms:
            for arm in group:
                validate_config("prefetch", arm["config"])
        self.assertEqual({a["config"]["seed"] for a in arms[0]}, {3101})
        self.assertEqual({a["config"]["seed"] for a in arms[1]}, {3102})

    def test_sp_diagnosis_matrix_preserves_original_gates_and_budget(self):
        from ditserve.evaluation.workflow_serving_sp import normalize_config
        api.initial_jobs(self.study)
        original_jobs = copy.deepcopy(self.study["jobs"])
        base = api.read(report.current_job(self.study, "sp-diagnostic")["input"])
        atomic_json(self.root / "budget.json", dict(charged_seconds=14508, active=None))
        budget_before = (self.root / "budget.json").read_bytes()
        controls = api.sp_diagnosis_jobs(self.study)
        self.assertEqual([j["id"] for j in controls], ["sp-native-control", "sp-linear-controls"])
        self.assertEqual(self.study["jobs"][:-2], original_jobs)
        self.assertEqual((self.root / "budget.json").read_bytes(), budget_before)
        self.assertEqual(self.study["budget_seconds"], 28800)
        expected = [
            (["cache_dit", "baseline"], False, [], 2900, 1600, 3000),
            (["full_rows_linear", "no_reduced_precision"], True,
             ["full_rows_linear", "no_reduced_precision"], 3500, 1800, 3600),
        ]
        for job, (variants, extended, localize, internal_timeout, estimate, timeout) in zip(controls, expected):
            config = api.read(job["input"])
            self.assertEqual(config, dict(base, variants=variants, extended_diagnostics=extended,
                                        localize_modes=localize, timeout_seconds=internal_timeout))
            self.assertEqual(normalize_config(config)["threshold"], .05)
            self.assertEqual(job["requires"], ["acceptance"])
            self.assertEqual(job["family"], "sp")
            self.assertEqual(job["devices"], [0, 1])
            self.assertEqual((job["estimated_seconds"], job["timeout_seconds"]), (estimate, timeout))
            self.assertEqual(api.digest(config), job["configuration_sha256"])
        self.assertEqual(api.sp_diagnosis_jobs(self.study), controls)
        self.assertEqual(len(self.study["jobs"]), len(original_jobs) + 2)
        original_diagnostic = next(job for job in original_jobs if job["id"] == "sp-diagnostic")
        self.assertEqual(report.current_job(self.study, "sp-diagnostic"), original_diagnostic)
        self.assertEqual(report.current_job(self.study, "elastic-pause")["requires"],
                         ["sp-diagnostic", "elastic-transfer"])
        control_ids = {job["id"] for job in controls}
        self.assertTrue(all(control_ids.isdisjoint(job["requires"]) for job in original_jobs))

    def test_sp_diagnosis_reuses_old_native_revision_but_binds_new_linear_revision(self):
        api.initial_jobs(self.study)
        base = api.read(report.current_job(self.study, "sp-diagnostic")["input"])
        self.study.update(source=str(self.root / "sources/r004"), source_digest="old-revision")
        old = api.add_job(self.study, "sp-native-control", "sp",
            dict(base, variants=["cache_dit", "baseline"], extended_diagnostics=False,
                 localize_modes=[], timeout_seconds=2900),
            module="workflow_serving_sp", requires=["acceptance"], estimate=1600, timeout=3000)
        old_before, input_before = copy.deepcopy(old), Path(old["input"]).read_bytes()
        self.study.update(source=str(self.root / "sources/r005"), source_digest="new-revision")
        native, linear = api.sp_diagnosis_jobs(self.study)
        self.assertEqual(native, old_before)
        self.assertEqual(Path(old["input"]).read_bytes(), input_before)
        self.assertEqual(linear["source"], self.study["source"])
        self.assertEqual(linear["source_digest"], "new-revision")
        changed = api.read(native["input"])
        changed["variants"] = ["cache_dit"]
        atomic_json(native["input"], changed)
        # Even a self-consistent new hash cannot silently replace the frozen arm.
        native["configuration_sha256"] = api.digest(changed)
        with self.assertRaisesRegex(ValueError, "silently replace"):
            api.sp_diagnosis_jobs(self.study)
        self.assertEqual(native["source"], old_before["source"])

    def test_sp_diagnosis_rejects_changed_original_input(self):
        api.initial_jobs(self.study)
        original = report.current_job(self.study, "sp-diagnostic")
        config = api.read(original["input"])
        config["threshold"] = .1
        atomic_json(original["input"], config)
        before = len(self.study["jobs"])
        with self.assertRaisesRegex(ValueError, "frozen input contents changed"):
            api.sp_diagnosis_jobs(self.study)
        self.assertEqual(len(self.study["jobs"]), before)

    def replay_sources(self):
        """Small immutable byte fixtures; generation does not load model tensors."""
        api.initial_jobs(self.study)
        unused, source = api.sp_diagnosis_jobs(self.study)
        output = Path(source["output"])
        output.mkdir(parents=True)
        result = output / "run.json"
        atomic_json(result, dict(schema="ditserve.workflow.serving.sp.v1",
                                status="qualification_failed", qualification=dict(passed=False)))
        launch = self.root / "launches" / source["id"] / "launch.json"
        launch.parent.mkdir(parents=True)
        atomic_json(launch, dict(status="completed", output_sha256=api.binding(result)["sha256"]))
        tensors = ("to_q_input", "to_q", "feed_forward_input", "feed_forward_first_projection",
                   "feed_forward_activation", "feed_forward_second_projection")
        for seed in (3501, 3502):
            for frames in (17, 33):
                for rank in (0, 1):
                    directory = output / f"rank{rank}/seed{seed}/f{frames}/localization/reference/step00/block00-qkv"
                    directory.mkdir(parents=True)
                    rows = []
                    for branch in ("cond", "uncond"):
                        for tensor in tensors:
                            key = f"{branch}.block00.{tensor}"
                            path = directory / (key + ".npy")
                            path.write_bytes(f"fixture {seed}/{frames}/{rank}/{key}".encode())
                            rows.append(dict(key=key, block=0, branch=branch, tensor=tensor,
                                             path=api.container_path(self.study, path),
                                             hash=dict(sha256="a" * 64)))
                    atomic_json(directory / "capture.json",
                                dict(status="completed", rank=rank, target_step=0, target_block=0,
                                     scheduler_steps=20, scope="selected_block_qkv",
                                     comparison_layout="SP2 local sequence vs matching SP1 contiguous sequence shard",
                                     layers=rows))
        self.study["model_root"] = str(self.root / "models")
        transformer = Path(self.study["model_root"]) / "wan2.1-t2v-1.3b/transformer"
        transformer.mkdir(parents=True)
        shard = "diffusion_pytorch_model-00001-of-00002.safetensors"
        (transformer / shard).write_bytes(b"fixture model tensor source")
        modules = ("blocks.0.attn1.to_q", "blocks.0.ffn.net.0.proj", "blocks.0.ffn.net.2")
        atomic_json(transformer / "diffusion_pytorch_model.safetensors.index.json",
                    dict(weight_map={module + "." + suffix: shard for module in modules
                                     for suffix in ("weight", "bias")}))
        api.save(self.study)
        return source

    def test_linear_replay_binds_complete_matrix_and_keeps_elastic_gate(self):
        from ditserve.evaluation.workflow_linear_replay import normalize_config
        source = self.replay_sources()
        old_jobs = copy.deepcopy(self.study["jobs"])
        job = api.sp_linear_replay_job(self.study)
        self.assertIsNotNone(job)
        config = api.read(job["input"])
        self.assertEqual(normalize_config(config), dict(config, save_arrays=False))
        self.assertEqual(job["devices"], [1])
        self.assertEqual(suite.selected_devices(job), [1])
        self.assertEqual(job["requires"], ["acceptance"])
        self.assertEqual((job["family"], job["estimated_seconds"], job["timeout_seconds"]), ("sp", 90, 360))
        self.assertEqual((config["device"], config["gpu_uuid"]), ("cuda:0", UUIDS[1]))
        self.assertEqual(config["cpu_affinity"], list(range(60, 68)))
        self.assertEqual(config["repeats"], 3)
        self.assertTrue(config["zero_padded"])
        self.assertEqual(config["operators"], ["to_q", "ffn.net.0.proj", "ffn.net.2"])
        self.assertEqual(config["branches"], ["cond", "uncond"])
        self.assertEqual([(case["seed"], case["frames"]) for case in config["cases"]],
                         [(3501, 17), (3501, 33), (3502, 17), (3502, 33)])
        for case in config["cases"]:
            self.assertEqual([capture["rank"] for capture in case["captures"]], [0, 1])
            for capture in case["captures"]:
                self.assertEqual(len(capture["files"]), 12)
                self.assertEqual(len(capture["sha256"]), 64)
                for row in capture["files"].values():
                    self.assertTrue(row["path"].startswith("/data/workflow/run03/measurements/sp-linear-controls/"))
                    self.assertEqual(len(row["sha256"]), 64)
        # Config, 8 manifests, 96 arrays, parent input/result, model index/shard.
        self.assertEqual(len(job["input_bindings"]), 109)
        self.assertEqual([api.binding(row["path"]) for row in job["input_bindings"]], job["input_bindings"])
        self.assertFalse(job["metadata"]["performance_evidence"])
        self.assertFalse(job["metadata"]["deployment_qualification"])
        self.assertEqual(job["metadata"]["source_job"], source["id"])
        self.assertEqual(len(job["metadata"]["model_tensors"]), 6)
        self.assertEqual(self.study["jobs"][:-1], old_jobs)
        self.assertEqual(report.current_job(self.study, "elastic-pause")["requires"],
                         ["sp-diagnostic", "elastic-transfer"])
        self.assertTrue(all(job["id"] not in row["requires"] for row in old_jobs))

    def test_linear_replay_reuses_frozen_job_and_rejects_changed_array(self):
        self.replay_sources()
        job = api.sp_linear_replay_job(self.study)
        before, input_before = copy.deepcopy(job), Path(job["input"]).read_bytes()
        self.study.update(source=str(self.root / "sources/r007"), source_digest="future-revision")
        self.assertEqual(api.sp_linear_replay_job(self.study), before)
        self.assertEqual(Path(job["input"]).read_bytes(), input_before)
        array = next(Path(row["path"]) for row in job["input_bindings"] if row["path"].endswith(".npy"))
        array.write_bytes(array.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "silently replace"):
            api.sp_linear_replay_job(self.study)
        self.assertEqual(job, before)
        self.assertEqual(Path(job["input"]).read_bytes(), input_before)

    def test_candidate_replay_is_new_bound_job_and_preserves_original(self):
        self.replay_sources()
        original = api.sp_linear_replay_job(self.study)
        old = copy.deepcopy(original)
        config_before = Path(original["input"]).read_bytes()
        options = dict(identifier="sp-linear-candidates", candidates=["global_tile_1024"],
                       microbenchmark=dict(enabled=True, warmups=3, repeats=10),
                       estimate=600, timeout=1200)
        candidate = api.sp_linear_replay_job(self.study, **options)
        self.assertEqual(api.sp_linear_replay_job(self.study, **options), candidate)
        self.assertEqual(len(candidate["input_bindings"]), 109)
        self.assertEqual(api.read(candidate["input"])["candidates"], ["global_tile_1024"])
        self.assertEqual(original, old)
        self.assertEqual(Path(original["input"]).read_bytes(), config_before)
        self.assertNotIn("candidates", api.read(original["input"]))

    def test_optimization_cli_rejects_ambiguous_modes_before_reading_study(self):
        invalid = [["--optimize-followup"],
                   ["--run", "--optimize-followup", "--diagnose-sp"],
                   ["--run", "--optimize-followup", "--only", "x"],
                   ["--run", "--optimize-followup", "--retry-block", "x"]]
        with patch.object(driver.api, "read", side_effect=AssertionError("must reject before reading")):
            for args in invalid:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        driver.main(args)
                    self.assertEqual(raised.exception.code, 2)
        for options in (dict(only="x"), dict(retry="x"), dict(diagnose_sp=True)):
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                driver.execute(self.study, optimize_followup=True, **options)

    def test_sp2_acceptance_rejects_ambiguous_modes_before_reading_study(self):
        invalid = [["--accept-sp2"], ["--run", "--accept-sp2", "--diagnose-sp"],
                   ["--run", "--accept-sp2", "--optimize-followup"],
                   ["--run", "--accept-sp2", "--only", "x"],
                   ["--run", "--accept-sp2", "--retry-block", "x"]]
        with patch.object(driver.api, "read", side_effect=AssertionError("must reject before reading")):
            for args in invalid:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        driver.main(args)
                    self.assertEqual(raised.exception.code, 2)
        for options in (dict(only="x"), dict(retry="x"), dict(diagnose_sp=True),
                        dict(optimize_followup=True)):
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                driver.execute(self.study, accept_sp2=True, **options)

    def test_sp2_acceptance_runs_only_new_chain_and_preserves_budget(self):
        from ditserve.evaluation import workflow_sp2_acceptance_study as acceptance
        self.study["source_revisions"] = [dict(source=str(driver.REPO), source_digest="fixture")]
        atomic_json(self.root / "budget.json", dict(charged_seconds=22442.93, active=None))
        budget_before = (self.root / "budget.json").read_bytes()
        events = []
        with patch.object(driver, "verify_revision", side_effect=lambda *a: events.append("verify")), \
             patch.object(driver, "helper", return_value=SimpleNamespace(recover_study=lambda *a, **k:
                          events.append("recover") or dict(status="settled"))), \
             patch.object(acceptance, "run_followup", side_effect=lambda *a: events.append("accept")) as run, \
             patch.object(driver.report, "build_report", side_effect=lambda *a, **k: events.append("report")), \
             patch.object(driver, "archive", side_effect=lambda *a: events.append("archive")), \
             patch.object(driver.subprocess, "run", side_effect=AssertionError("CPU fixture cannot launch")):
            driver.execute(self.study, accept_sp2=True)
        self.assertEqual(events, ["verify", "recover", "accept", "report", "archive"])
        run.assert_called_once_with(self.study, driver.run_job)
        self.assertEqual(self.study["status"], "settled")
        self.assertIsNone(self.study["active_job"])
        self.assertEqual((self.root / "budget.json").read_bytes(), budget_before)

    def test_linear_replay_skips_incomplete_or_unverified_source_without_new_gate(self):
        source = self.replay_sources()
        before = copy.deepcopy(self.study["jobs"])
        launch = self.root / "launches" / source["id"] / "launch.json"
        ledger = api.read(launch)
        for changed in (dict(ledger, status="running"), dict(ledger, output_sha256="0" * 64)):
            atomic_json(launch, changed)
            self.assertIsNone(api.sp_linear_replay_job(self.study))
        atomic_json(launch, ledger)
        capture = next(Path(source["output"]).glob("rank*/seed*/f*/localization/reference/step00/block00-qkv/capture.json"))
        manifest = api.read(capture)
        atomic_json(capture, dict(manifest, status="running"))
        self.assertIsNone(api.sp_linear_replay_job(self.study))
        atomic_json(capture, manifest)
        missing = capture.parent / (manifest["layers"][0]["key"] + ".npy")
        missing.unlink()
        self.assertIsNone(api.sp_linear_replay_job(self.study))
        self.assertEqual(self.study["jobs"], before)
        self.assertFalse((self.root / "inputs/sp-linear-replay.json").exists())
        self.assertEqual(self.study["decisions"][-1]["decision"], "diagnostic_source_unavailable")
        self.assertIn(str(missing), self.study["decisions"][-1]["reason"])

    def test_diagnose_sp_cli_requires_run_and_excludes_only_or_retry(self):
        invalid = [["--diagnose-sp"], ["--run", "--diagnose-sp", "--only", "sp-native-control"],
                   ["--run", "--diagnose-sp", "--retry-block", "sp-native-control"]]
        with patch.object(driver.api, "read", side_effect=AssertionError("must reject before reading study")):
            for arguments in invalid:
                with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        driver.main(arguments)
                    self.assertEqual(raised.exception.code, 2)
        for options in (dict(only="x"), dict(retry="x")):
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                driver.execute(self.study, diagnose_sp=True, **options)

    def test_diagnose_sp_executes_only_controls_after_recovery_and_archives(self):
        api.initial_jobs(self.study)
        revision = dict(source=str(driver.REPO), source_digest="fixture")
        self.study["source_revisions"] = [revision]
        self.study.update(status="settled", finished_unix=123, error="old attempt")
        atomic_json(self.root / "budget.json", dict(charged_seconds=14508, active=None))
        budget_before = (self.root / "budget.json").read_bytes()
        events = []
        recover = Mock(side_effect=lambda *a, **k: events.append("recover") or dict(status="settled"))
        def run_control(study, job):
            self.assertEqual(study["status"], "running")
            self.assertNotIn("finished_unix", study)
            self.assertNotIn("error", study)
            events.append(job["id"])
            return False
        with patch.object(driver, "verify_revision", side_effect=lambda *a: events.append("verify")), \
             patch.object(driver, "helper", return_value=SimpleNamespace(recover_study=recover)), \
             patch.object(driver, "run_job", side_effect=run_control), \
             patch.object(driver.api, "sp_linear_replay_job",
                          side_effect=lambda s: dict(id="sp-linear-replay")) as replay, \
             patch.object(driver.report, "build_report", side_effect=lambda *a, **k: events.append("report")) as build, \
             patch.object(driver, "archive", side_effect=lambda *a: events.append("archive")) as archive, \
             patch.object(driver.subprocess, "run", side_effect=AssertionError("CPU fixture cannot launch")):
            driver.execute(self.study, diagnose_sp=True)
        self.assertEqual(events, ["verify", "recover", "sp-native-control", "sp-linear-controls",
                                  "sp-linear-replay", "report", "archive"])
        replay.assert_called_once_with(self.study)
        self.assertTrue(recover.call_args.kwargs["driver_lock_held"])
        build.assert_called_once_with(self.root, final=True)
        archive.assert_called_once_with(self.study)
        self.assertEqual(self.study["status"], "settled")
        self.assertIsNone(self.study["active_job"])
        self.assertGreater(self.study["finished_unix"], 123)
        self.assertNotIn("error", self.study)
        self.assertEqual((self.root / "budget.json").read_bytes(), budget_before)

    def test_diagnose_sp_unavailable_replay_still_finishes_report(self):
        api.initial_jobs(self.study)
        self.study["source_revisions"] = [dict(source=str(driver.REPO), source_digest="fixture")]
        with patch.object(driver, "verify_revision"), \
             patch.object(driver, "helper", return_value=SimpleNamespace(
                 recover_study=lambda *a, **k: dict(status="settled"))), \
             patch.object(driver, "run_job", return_value=False) as run, \
             patch.object(driver.report, "build_report") as build, \
             patch.object(driver, "archive") as archive:
            driver.execute(self.study, diagnose_sp=True)
        self.assertEqual([call.args[1]["id"] for call in run.call_args_list],
                         ["sp-native-control", "sp-linear-controls"])
        self.assertFalse(any(job["id"] == "sp-linear-replay" for job in self.study["jobs"]))
        self.assertEqual(self.study["decisions"][-1]["decision"], "diagnostic_source_unavailable")
        build.assert_called_once_with(self.root, final=True)
        archive.assert_called_once_with(self.study)

    def test_core_contrast_and_quantum_screen_use_selected_controls(self):
        self.generate()
        core = api.read(report.current_job(self.study, "cpu-online-cores-r0")["input"])
        self.assertEqual([p["text_prefetch_depth"] for p in core["phases"]], [2, 2])
        self.assertEqual([p["worker_overrides"]["wan0"]["cpu_threads"] for p in core["phases"]], [8, 16])
        for logical in ("quantum-screen", "mixed-poisson-s3301", "elastic-online"):
            config = api.read(report.current_job(self.study, logical)["input"])
            self.assertTrue(all(p["text_prefetch_depth"] == 2 for p in config["phases"]))
            self.assertTrue(all(p["worker_overrides"]["wan0"]["cpu_threads"] == 16 for p in config["phases"]))

    def test_retry_keeps_prior_inputs_budget_and_profiler_namespace(self):
        self.generate()
        old = copy.deepcopy(report.current_job(self.study, "overlap-trace"))
        before = Path(old["input"]).read_bytes()
        atomic_json(self.root / "budget.json", dict(charged_seconds=123, active=None))
        new = api.retry_block(self.study, old["id"])
        self.assertNotEqual(new["output"], old["output"])
        self.assertNotEqual(new["input"], old["input"])
        self.assertEqual(Path(old["input"]).read_bytes(), before)
        self.assertEqual(api.read(self.root / "budget.json")["charged_seconds"], 123)
        self.assertTrue(new["profiling"])
        self.assertIn("/traces/" + new["id"], " ".join(new["command"]))
        api.extension_jobs(self.study)
        self.assertEqual(report.current_job(self.study, "overlap-trace")["id"], new["id"])
        self.assertEqual(len([j for j in self.study["jobs"] if j["logical_id"] == "overlap-trace"]), 2)

    def test_active_retry_refused_and_family_leftover_rolls_forward(self):
        api.initial_jobs(self.study)
        atomic_json(self.root / "budget.json", dict(charged_seconds=1000,
            active=None, attempts=[dict(job_id="acceptance", charged_seconds=1000)]))
        self.assertEqual(driver.family_available(self.study, "cpu"), 5300)
        self.assertEqual(driver.remaining(self.study), 27800)
        self.study["active_job"] = "acceptance"
        with self.assertRaisesRegex(RuntimeError, "active"):
            api.retry_block(self.study, "acceptance")

    def test_resume_rejects_changed_frozen_control_selection(self):
        self.generate()
        before = Path(report.current_job(self.study, "mixed-poisson-s3301")["input"]).read_bytes()
        self.study["selections"]["cpu"]["cores"] = 8
        with self.assertRaisesRegex(ValueError, "silently replace"):
            api.mixed_jobs(self.study, self.profile, self.capacities["mixed"], 300)
        self.assertEqual(Path(report.current_job(self.study, "mixed-poisson-s3301")["input"]).read_bytes(), before)

    def test_budget_and_dependency_deferral_do_not_launch_processes(self):
        api.initial_jobs(self.study)
        with patch.object(driver.subprocess, "run", side_effect=AssertionError("must not launch")):
            self.assertFalse(driver.run_job(self.study, report.current_job(self.study, "cost")))
            self.assertEqual(report.current_job(self.study, "cost")["decision"], "prerequisite_not_qualified")
            atomic_json(self.root / "budget.json", dict(charged_seconds=28750, active=None))
            self.assertFalse(driver.run_job(self.study, report.current_job(self.study, "acceptance")))
            self.assertEqual(report.current_job(self.study, "acceptance")["decision"], "complete_block_budget_deferred")

    def test_native_environment_rejects_ordinal_uuid_disagreement(self):
        api.initial_jobs(self.study)
        job = copy.deepcopy(report.current_job(self.study, "cost"))
        job["gpu_mapping"][0]["uuid"] = "GPU-changed"
        with self.assertRaisesRegex(ValueError, "UUIDs"):
            suite.selected_devices(job)

    def test_generated_native_configs_resolve_without_gpu_execution(self):
        core = CoreBridge()
        if not core.available():
            self.skipTest("build the native core first")
        self.generate()
        inventory = [dict(index=i, uuid=UUIDS[i], name="NVIDIA GeForce RTX 4090",
                          memory_mib=24564) for i in range(2)]
        for job in self.study["jobs"]:
            if job.get("command_type") != "native":
                continue
            value = core.call("resolve", dict(config=api.read(job["input"]),
                env=job["environment_input"], gpu_inventory=inventory))
            self.assertNotIn("error", value, (job["id"], value))
            self.assertEqual(value["schema"], "ditserve.study.v1")


if __name__ == "__main__":
    unittest.main()
