"""CPU fixtures for the append-only dispatch-refresh experimental controls."""

import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_dispatch_refresh_study as refresh
from ditserve.evaluation import workflow_encoding_optimization_study as encoding
from ditserve.evaluation import workflow_next_study as api


class DispatchRefreshStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "run03"
        self.repo = Path(__file__).resolve().parents[2]
        self.profile = api.read(self.repo / "configs/profiles/rtx4090-sp1-local.json")
        for frames in (17, 33):
            self.profile["classes"][f"wan1.3b-480x832x{frames}"]["steps"] = 20
        self.cells = [dict(status="completed", elapsed_ms=100000, requests=[
            dict(kind="video", status="completed", start_ms=1000 + i * 100,
                 finish_ms=50000 + i * 1000, spec=dict(frames=33)) for i in range(2)]) for _ in range(3)]
        self.anchors = refresh.contention._anchors(self.profile, self.cells)
        self.rows = refresh.contention.contention_trace(self.profile, 4301, self.anchors)
        revisions = []
        for number in (13, 14):
            source = self.root / "sources" / f"r{number:03}"
            for name, text in (("bin/ditserve", "cpu fixture"),
                               ("workers/ditserve/models/wan/backend.py", "unchanged model"),
                               ("workers/ditserve/runtime/worker.py", "unchanged worker"),
                               ("src/ditserve/control/study.cpp", f"controller revision {number}")):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            (source / "configs/env").mkdir(parents=True)
            shutil.copyfile(self.repo / "configs/env/rtx4090-2x24g.json",
                            source / "configs/env/rtx4090-2x24g.json")
            revision = api.seal(source)
            api.atomic_json(source.parent / (source.name + ".binding.json"), revision)
            revisions.append(revision)
        self.study = dict(run_id="run03", root=str(self.root), source=revisions[-1]["source"],
            source_digest=revisions[-1]["source_digest"], source_revisions=revisions, jobs=[],
            active_job=None, budget_seconds=None, budget_mode="unlimited", family_caps_seconds=copy.deepcopy(api.CAPS),
            model_bindings=[dict(path="/models/wan/model.bin", bytes=128, mtime_ns=1)],
            selections=dict(cpu=dict(cores=8, depth=1)),
            gpu_mapping=[dict(ordinal=i, uuid=f"GPU-00000000-0000-0000-0000-00000000000{i}") for i in (0, 1)])
        api.atomic_json(self.root / "profile.json", self.profile)
        api.atomic_json(self.root / "calibration.json", dict(profile=self.profile))
        self.evidence = {}
        for logical in refresh.BASE_PREREQUISITES + refresh.HISTORICAL_JOBS:
            old = revisions[0]
            metadata = {}
            config = dict(fixture=logical)
            result = dict(status="completed")
            if logical == "calibration-video":
                result["phases"] = [dict(id=str(index), cells=[cell]) for index, cell in enumerate(self.cells)]
            if logical in refresh.HISTORICAL_JOBS:
                index = refresh.HISTORICAL_JOBS.index(logical)
                config = encoding.native_online_config(self.profile, self.rows,
                    optimization="wan_negative_cache", prompt_distribution="mixed_negative", repeat=index)
                metadata = dict(role="encoding_online", optimization="wan_negative_cache", seed=encoding.WAN_SEEDS[index])
            input_path = self.root / "inputs" / (logical + ".json")
            output_path = self.root / "measurements" / logical / "run.json"
            launch_path = self.root / "launches" / logical / "launch.json"
            api.atomic_json(input_path, config)
            api.atomic_json(output_path, result)
            launch = dict(status="completed", wall_seconds=310.)
            api.atomic_json(launch_path, launch)
            self.study["jobs"].append(dict(id=logical, logical_id=logical, family="extension",
                source=old["source"], source_digest=old["source_digest"], input=str(input_path),
                configuration_sha256=api.digest(config), input_bindings=[api.binding(input_path)], metadata=metadata))
            self.evidence[logical] = dict(guarded=True, qualified=True, capacity=True,
                path=output_path, result=result, launch=launch)
        api.save(self.study)
        api.atomic_json(self.root / "budget.json", dict(limit_seconds=None, charged_seconds=31000))
        self.patcher = patch.object(refresh.report, "evidence", side_effect=lambda study, job: self.evidence[job["id"]])
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_three_arms_vary_only_cache_and_explicit_refresh(self):
        before = copy.deepcopy((self.profile, self.rows))
        for repeat in range(3):
            config = refresh.native_config(self.profile, self.rows, repeat=repeat)
            self.assertTrue(refresh.validate_config(config, repeat=repeat))
            self.assertEqual([p["id"] for p in config["phases"]], list(refresh.ARMS[repeat:] + refresh.ARMS[:repeat]))
            normalized = []
            for phase in config["phases"]:
                self.assertIs(phase["dispatch_refresh"], phase["id"] == "cache_on_refresh")
                self.assertEqual(phase["text_prefetch_depth"], 1)
                self.assertEqual(phase["options"]["quantum_ms"], 300)
                self.assertEqual(phase["options"]["decode_frames"], 1)
                self.assertEqual(phase["controller_cpu_affinity"], [59])
                self.assertEqual(len(phase["workload"]), 24)
                item = copy.deepcopy(phase)
                enabled = refresh.CACHE_ENABLED[item.pop("id")]
                item.pop("dispatch_refresh")
                for rank in (0, 1):
                    wan = item["worker_overrides"][f"wan{rank}"]
                    self.assertEqual(wan["cpu_threads"], 8)
                    self.assertEqual(wan["cpu_affinity"], list(range(rank * 60, rank * 60 + 8)))
                    self.assertEqual(item["worker_overrides"][f"sana{rank}"]["options"], {})
                    for key, value in encoding.WAN_CACHE.items():
                        self.assertEqual(wan["options"].pop(key, None), value if enabled else None)
                normalized.append(item)
            self.assertEqual(normalized, [normalized[0]] * 3)
            for frames in (17, 33):
                prompts = {r["spec"]["negative_prompt"] for r in normalized[0]["workload"]
                           if r["kind"] == "video" and r["spec"]["frames"] == frames}
                self.assertEqual(prompts, {"", encoding.NEGATIVE_PROMPT})
        self.assertEqual((self.profile, self.rows), before)

    def test_strict_validation_rejects_intervention_or_control_drift(self):
        config = refresh.native_config(self.profile, self.rows)
        changes = [lambda x: x["phases"][0].pop("dispatch_refresh"),
                   lambda x: x["phases"][0].update(dispatch_refresh=0),
                   lambda x: x["phases"][2].update(dispatch_refresh=False),
                   lambda x: x["phases"][0].update(text_prefetch_depth=2),
                   lambda x: x["phases"][0]["options"].update(quantum_ms=100),
                   lambda x: x["phases"][0]["options"].update(decode_frames=True),
                   lambda x: x["phases"][0]["worker_overrides"]["wan0"].update(cpu_threads=4),
                   lambda x: x["phases"][1]["workload"][0].update(arrival_ms=999),
                   lambda x: x["phases"].reverse(),
                   lambda x: x.update(unrecognized_option=True)]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                altered = copy.deepcopy(config)
                change(altered)
                with self.assertRaises(ValueError):
                    refresh.validate_config(altered, repeat=0)

    def test_read_only_plan_separates_screen_and_three_new_trajectories(self):
        before, ledger = copy.deepcopy(self.study), (self.root / "budget.json").read_bytes()
        plan = refresh.build_plan(self.study)
        self.assertEqual(self.study, before)
        self.assertEqual((self.root / "budget.json").read_bytes(), ledger)
        self.assertFalse(plan["ready_for_gpu"])
        self.assertEqual([job["metadata"]["seed"] for job in plan["jobs"]], [4300, 4301, 4302, 4303])
        self.assertEqual([job["metadata"]["repeat"] for job in plan["jobs"]], [0, 0, 1, 2])
        self.assertEqual([job["metadata"]["stage"] for job in plan["jobs"]], ["screening"] + ["confirmation"] * 3)
        self.assertEqual(len({job["metadata"]["trace_sha256"] for job in plan["jobs"]}), 4)
        for job in plan["jobs"]:
            self.assertEqual(job["metadata"]["monitor_cpu_affinity"], [58])
            self.assertEqual(job["estimated_seconds"], 560)
            self.assertEqual(job["timeout_seconds"], 1800)
            self.assertTrue(set(refresh.BASE_PREREQUISITES).issubset(job["requires"]))
            self.assertEqual(refresh.job_id(4300) in job["requires"], job["metadata"]["stage"] == "confirmation")
        paths = {row["path"] for row in plan["input_bindings"]}
        self.assertNotIn(str(self.root / "study.json"), paths)
        self.assertNotIn(str(self.root / "budget.json"), paths)
        self.assertTrue({str(self.root / "sources" / (name + ".binding.json")) for name in ("r013", "r014")} <= paths)

    def test_register_is_append_only_idempotent_and_preserves_budget(self):
        before = copy.deepcopy(self.study["jobs"])
        ledger = (self.root / "budget.json").read_bytes()
        jobs = refresh.register(self.study)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(self.study["jobs"][:len(before)], before)
        self.assertEqual((self.root / "budget.json").read_bytes(), ledger)
        for job in jobs:
            self.assertEqual(job["source"], self.study["source"])
            self.assertEqual(job["source_digest"], self.study["source_digest"])
            self.assertEqual(job["command_type"], "native")
            self.assertEqual(job["devices"], [0, 1])
            self.assertTrue(refresh.validate_config(api.read(job["input"]), repeat=job["metadata"]["repeat"]))
            self.assertIn("--keep-raw", job["command"])
        frozen = copy.deepcopy(self.study)
        self.assertEqual(refresh.register(self.study), jobs)
        self.assertEqual(self.study, frozen)

    def test_preflight_checks_all_existing_jobs_before_appending_or_changing(self):
        jobs = refresh.register(self.study)
        jobs[-1]["metadata"]["repeat"] = 0
        before = copy.deepcopy(self.study)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            refresh.register(self.study)
        self.assertEqual(self.study, before)

    def test_resume_rejects_changed_command_and_gpu_identity(self):
        jobs = refresh.register(self.study)
        original = copy.deepcopy(jobs[-1])
        for key, value in (("command", jobs[-1]["command"][:-2]),
                           ("gpu_mapping", list(reversed(jobs[-1]["gpu_mapping"])))):
            with self.subTest(key=key):
                jobs[-1].update(copy.deepcopy(original))
                jobs[-1][key] = value
                before = copy.deepcopy(self.study)
                with self.assertRaisesRegex(ValueError, "cannot replace"):
                    refresh.register(self.study)
                self.assertEqual(self.study, before)

    def test_failed_numeric_or_historical_binding_prevents_registration(self):
        before = copy.deepcopy(self.study)
        logical = refresh.NUMERIC_PREREQUISITES[0]
        self.evidence[logical]["qualified"] = False
        with self.assertRaisesRegex(ValueError, "guarded numerical"):
            refresh.register(self.study)
        self.assertEqual(self.study, before)
        self.evidence[logical]["qualified"] = True
        Path(self.study["jobs"][0]["input"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "historical input"):
            refresh.register(self.study)
        self.assertEqual(self.study, before)

    def test_model_source_change_requires_new_numeric_even_after_new_seal(self):
        source = Path(self.study["source"])
        (source / "workers/ditserve/models/wan/backend.py").write_text("changed model")
        revision = api.seal(source)
        self.study.update(source_digest=revision["source_digest"])
        self.study["source_revisions"][-1] = revision
        api.atomic_json(source.parent / (source.name + ".binding.json"), revision)
        with self.assertRaisesRegex(ValueError, "new numerical qualification"):
            refresh.register(self.study)
        self.assertEqual(len(self.study["jobs"]), 8)

    def test_active_old_revision_or_reused_seed_does_not_register(self):
        self.study["active_job"] = "running"
        with self.assertRaisesRegex(ValueError, "active"):
            refresh.register(self.study)
        self.study["active_job"] = None
        old = self.study["source_revisions"][0]
        self.study.update(source=old["source"], source_digest=old["source_digest"])
        self.assertEqual(len(refresh.build_plan(self.study)["jobs"]), 4)
        with self.assertRaisesRegex(ValueError, "freeze a new"):
            refresh.register(self.study)
        self.study["jobs"][0]["metadata"]["seed"] = 4301
        with self.assertRaisesRegex(ValueError, "reuse historical"):
            refresh.build_plan(self.study)


if __name__ == "__main__":
    unittest.main()
