"""CPU proposal export must preserve historical evidence and its budget."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_next_study as api

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("encoding_plan_cli", REPO / "docker/plan-encoding-optimization.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class EncodingPlanCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root, self.output = self.directory / "history", self.directory / "proposal"
        self.root.mkdir()
        self.profile = api.read(REPO / "configs/profiles/rtx4090-sp1-local.json")
        for row in self.profile["classes"].values():
            if row["kind"] == "video":
                row["steps"] = 20
        self.source = self.directory / "source"
        environment_path = self.source / "configs/env/rtx4090-2x24g.json"
        environment_path.parent.mkdir(parents=True)
        api.atomic_json(environment_path, dict(name="fixture"))
        calibration_output = self.root / "measurements/calibration-video"
        calibration_output.mkdir(parents=True)
        api.atomic_json(calibration_output / "run.json", dict(status="completed"))
        launch = self.root / "launches/calibration-video/launch.json"
        launch.parent.mkdir(parents=True)
        api.atomic_json(launch, dict(status="completed"))
        api.atomic_json(self.root / "study.json", dict(root=str(self.root), active_job=None,
            source=str(self.source), run_id="fixture", budget_seconds=28800,
            family_caps_seconds=api.CAPS,
            gpu_mapping=[dict(uuid="GPU-00000000-0000-0000-0000-00000000000" + str(i)) for i in (0, 1)],
            jobs=[dict(id="calibration-video", output=str(calibration_output))]))
        api.atomic_json(self.root / "budget.json", dict(limit_seconds=28800, charged_seconds=27403.253727,
                                                       active=None, attempts=[]))
        api.atomic_json(self.root / "profile.json", self.profile)
        api.atomic_json(self.root / "calibration.json", dict(profile=self.profile))
        self.cells = [dict(requests=[
            dict(kind="video", status="completed", spec=dict(frames=33), start_ms=1000, finish_ms=40000),
            dict(kind="video", status="completed", spec=dict(frames=33), start_ms=1100, finish_ms=41000)])] * 3

    def export(self):
        with patch.object(cli.contention, "_calibration", return_value=({}, self.cells, [])), \
             patch.object(cli.api, "environment", return_value=dict(name="fixture")):
            return cli.prepare(self.root, self.output)

    def test_export_is_cpu_only_preserves_history_and_binds_nine_native_configs(self):
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = self.export()
        self.assertEqual({str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)
        self.assertTrue(result["proposal_only"])
        self.assertTrue(result["source_freeze_required"])
        self.assertFalse(result["historical_study_modified"])
        self.assertEqual(result["gpu_jobs_launched"], 0)
        self.assertEqual(len(result["native_configs"]), 9)
        self.assertEqual(len(result["qualification_configs"]), 4)
        self.assertEqual(len(result["environment_bindings"]), 2)
        for binding in result["environment_bindings"]:
            self.assertEqual(binding, api.binding(binding["path"]))
        for row in result["qualification_configs"]:
            config = api.read(row["binding"]["path"])
            self.assertEqual(row["configuration_sha256"], api.digest(config))
            self.assertEqual(row["binding"], api.binding(row["binding"]["path"]))
            self.assertTrue(row["requires_guarded_launcher"])
            self.assertEqual(config["device"], "cuda:0")
            self.assertEqual(config["cpu_mask"], list(range(60, 68)))
            self.assertEqual(row["qualification"], "pending")
        self.assertEqual(api.read(self.output / "plan.json"), result)
        for row in result["native_configs"]:
            config = api.read(row["binding"]["path"])
            self.assertEqual(row["binding"], api.binding(row["binding"]["path"]))
            self.assertEqual(row["configuration_sha256"], api.digest(config))
            self.assertEqual(row["qualification"], "pending")
            self.assertTrue(all(phase["output_mode"] == "check" for phase in config["phases"]))
        self.assertIn("workers/ditserve/models/wan/prompt_cache.py",
                      " ".join(row["path"] for row in result["development_sources"]))

    def test_output_must_be_new_and_outside_historical_study(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            cli.prepare(self.root, self.root / "new-proposal")
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_text("untouched")
        with self.assertRaises(FileExistsError):
            cli.prepare(self.root, self.output)
        self.assertEqual(sentinel.read_text(), "untouched")

    def test_active_budget_and_root_mismatch_rejected_without_output(self):
        budget = api.read(self.root / "budget.json")
        budget["active"] = dict(job_id="active")
        api.atomic_json(self.root / "budget.json", budget)
        with self.assertRaisesRegex(ValueError, "active study"):
            self.export()
        budget["active"] = None
        api.atomic_json(self.root / "budget.json", budget)
        api.atomic_json(self.root / "study.json", dict(root=str(self.directory / "other")))
        with self.assertRaisesRegex(ValueError, "root does not match"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_budget_changes_during_read_are_detected_before_publishing(self):
        def changed(*unused):
            budget = api.read(self.root / "budget.json")
            budget["charged_seconds"] += 1
            api.atomic_json(self.root / "budget.json", budget)
            return {}, self.cells, []
        with patch.object(cli.contention, "_calibration", side_effect=changed), \
             patch.object(cli.api, "environment", return_value=dict(name="fixture")):
            with self.assertRaisesRegex(RuntimeError, "changed while preparing"):
                cli.prepare(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_mutation_between_initial_binding_and_read_is_detected(self):
        original_read = api.read
        changed = False

        def mutate_after_read(path, default=None):
            nonlocal changed
            value = original_read(path, default)
            if Path(path) == self.root / "budget.json" and not changed:
                changed = True
                api.atomic_json(path, dict(value, charged_seconds=value["charged_seconds"] + 1))
            return value

        with patch.object(cli.api, "read", side_effect=mutate_after_read):
            with self.assertRaisesRegex(RuntimeError, "changed while preparing"):
                self.export()
        self.assertFalse(self.output.exists())

    def test_calibration_input_mutation_is_detected(self):
        def changed(*unused):
            api.atomic_json(self.root / "calibration.json", dict(profile=self.profile, changed=True))
            return {}, self.cells, []

        with patch.object(cli.contention, "_calibration", side_effect=changed), \
             patch.object(cli.api, "environment", return_value=dict(name="fixture")):
            with self.assertRaisesRegex(RuntimeError, "changed while preparing"):
                cli.prepare(self.root, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
