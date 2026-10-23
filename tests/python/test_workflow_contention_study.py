"""Frozen contention arrivals, observed triggers and guarded ablation planning."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_contention_study as contention
from ditserve.evaluation import workflow_next_report as report
from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation.workflow_budget import atomic_json

REPO = Path(__file__).resolve().parents[2]


class ContentionStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = dict(root=str(self.root), run_id="run03", source=str(REPO),
            source_digest="fixture", source_revisions=[], budget_seconds=28800,
            family_caps_seconds=api.CAPS, image="sha256:fixture", jobs=[], decisions=[], selections={},
            gpu_mapping=[dict(ordinal=i, uuid=f"GPU-00000000-0000-0000-0000-{i:012}",
                              name="RTX 4090") for i in range(2)])
        self.profile = api.read(REPO / "configs/profiles/rtx4090-sp1-local.json")
        for frames, encode, step, finish in ((17, 10242., 582., 1968.), (33, 9575., 1347., 3860.)):
            self.profile["classes"][f"wan1.3b-480x832x{frames}"].update(
                steps=20, encode_ms=encode, step_compute_ms=step, finish_ms=finish)
        self.profile["classes"]["sana-1024"]["service_ms"] = 233.
        self.capacities = dict(video=.07046, mixed=.3136)
        self.calibrations()

    def calibrations(self):
        for label in ("video", "mixed"):
            job = api.add_job(self.study, "calibration-" + label, "calibration", {}, module="fixture")
            videos = [dict(request_id=f"v{i}", kind="video", status="completed", spec=dict(frames=33),
                           start_ms=13610. if i == 0 else 11490.,
                           finish_ms=45340. if i == 0 else 43010.) for i in range(4)]
            images = [dict(request_id=f"i{i}", kind="image", status="completed") for i in range(16)]
            cells = [dict(id=f"repeat{i}", status="completed", cells=[dict(status="completed",
                        elapsed_ms=64000. if label == "mixed" else 114000.,
                        requests=copy.deepcopy(videos + (images if label == "mixed" else [])))])
                     for i in range(3)]
            self.result(job, dict(schema="ditserve.run.v1", status="completed", phases=cells),
                        462. if label == "mixed" else 612.)

    def result(self, job, result, wall):
        path = Path(job["output"]) / "run.json"
        atomic_json(path, result)
        launch = dict(status="completed", output_sha256=api.binding(path)["sha256"], wall_seconds=wall,
                      gpu_monitor=dict(errors=[], conflicts=[], memory={str(i): dict(samples=3,
                          minimum_free_bytes=3 * 1024 ** 3, total_bytes=24 * 1024 ** 3) for i in range(2)}))
        atomic_json(self.root / "launches" / job["id"] / "launch.json", launch)
        return path

    def test_three_rotated_arms_freeze_arrivals_and_bind_evidence(self):
        before = copy.deepcopy(self.study["jobs"])
        jobs = contention.contention_jobs(self.study, self.profile, self.capacities)
        self.assertEqual(self.study["jobs"][:2], before)
        self.assertEqual([j["id"] for j in jobs], [f"contention-s{s}" for s in contention.SEEDS])
        orders, traces = [], []
        for job in jobs:
            config = api.read(job["input"])
            phases = config["phases"]
            orders.append([p["id"] for p in phases])
            rows = phases[0]["workload"]
            traces.append(rows)
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(row["kind"] == "image" for row in rows), 20)
            self.assertEqual([r["spec"]["frames"] for r in rows if r["kind"] == "video"], [33, 33, 17, 17])
            self.assertTrue(all(p["workload"] == rows for p in phases))
            self.assertTrue(all(p["record_dispatch_timeline"] is True for p in phases))
            self.assertTrue(all(p["text_prefetch_depth"] == 1 for p in phases))
            self.assertTrue(all(p["controller_cpu_affinity"] == [59] for p in phases))
            self.assertEqual(job["metadata"]["trace_sha256"], api.digest(rows))
            self.assertEqual(job["metadata"]["comparison"], "contention")
            self.assertEqual(job["family"], "mixed")
            self.assertEqual(job["timeout_seconds"], 1000)
            self.assertGreater(job["estimated_seconds"], 450)
            self.assertLessEqual(job["estimated_seconds"], 600)
            self.assertEqual(len(job["input_bindings"]), 6)
            self.assertEqual([api.binding(b["path"]) for b in job["input_bindings"]], job["input_bindings"])
            self.assertNotIn("real_sp", phases[0])
        self.assertEqual(orders, [["step", "block100", "block300"],
                                  ["block100", "block300", "step"],
                                  ["block300", "step", "block100"]])
        self.assertNotEqual([r["arrival_ms"] for r in traces[0]], [r["arrival_ms"] for r in traces[1]])
        self.assertEqual(contention.contention_jobs(self.study, self.profile, self.capacities), jobs)
        self.assertEqual(len(self.study["jobs"]), 5)

    def test_unqualified_capacity_and_unmeasured_estimate_cannot_register(self):
        job = report.current_job(self.study, "calibration-video")
        launch_path = self.root / "launches" / job["id"] / "launch.json"
        launch = api.read(launch_path)
        launch["gpu_monitor"]["memory"]["1"]["minimum_free_bytes"] = 0
        atomic_json(launch_path, launch)
        with self.assertRaisesRegex(ValueError, "qualified calibration"):
            contention.contention_jobs(self.study, self.profile, self.capacities)
        self.assertEqual(len(self.study["jobs"]), 2)
        launch["gpu_monitor"]["memory"]["1"]["minimum_free_bytes"] = 3 * 1024 ** 3
        atomic_json(launch_path, launch)
        mixed = report.current_job(self.study, "calibration-mixed")
        path = self.root / "launches" / mixed["id"] / "launch.json"
        launch = api.read(path)
        launch["wall_seconds"] = 1000
        atomic_json(path, launch)
        with self.assertRaisesRegex(ValueError, "600-second"):
            contention.contention_jobs(self.study, self.profile, self.capacities)
        self.assertEqual(len(self.study["jobs"]), 2)

    def test_frozen_sources_cannot_be_rebound_and_cpu_recipe_is_fixed(self):
        contention.contention_jobs(self.study, self.profile, self.capacities)
        path = self.root / "launches/calibration-mixed/launch.json"
        launch = api.read(path)
        launch["audit_note"] = "changed"
        atomic_json(path, launch)
        with self.assertRaisesRegex(ValueError, "frozen contention source"):
            contention.contention_jobs(self.study, self.profile, self.capacities)
        with self.assertRaisesRegex(ValueError, "eight-core"):
            contention.contention_jobs(self.study, self.profile, self.capacities, cpu=dict(cores=16, depth=1))

    @staticmethod
    def observed_result():
        def event(start, finish, quantum, reason="block_preempt", group=None):
            return dict(start_ms=start, finish_ms=finish, quantum=quantum, reason=reason,
                        group=[0] if group is None else group)
        rows = [dict(request_id="v", kind="video", start_ms=10., arrival_ms=0., deadline_ms=100.,
                     dispatch_timeline=[event(10, 50, "step", "static_sp1"),
                                        event(50, 70, "finish", "decode")])]
        for identifier, arrival, start, deadline, group in (
                ("i0", 10., 30., 20., [0]), ("i1", 51., 65., 60., [0]),
                ("i2", 52., 66., 60., [0]), ("i3", 51., 65., 100., [1])):
            rows.append(dict(request_id=identifier, kind="image", arrival_ms=arrival,
                             start_ms=start, deadline_ms=deadline,
                             dispatch_timeline=[event(start, start + 1, "image", group=group)]))
        return dict(config=dict(phases=[dict(id="step", profile=dict(classes={"sana-1024": dict(service_ms=10.)}))]),
                    phases=[dict(id="step", cells=[dict(requests=rows)])])

    def test_triggers_use_actual_wait_and_same_gpu_host_windows(self):
        result = self.observed_result()
        values = contention.trigger_summary(result)["arms"]["step"]
        self.assertTrue(values["timeline_available"])
        self.assertTrue(values["pressure_triggered"])
        self.assertTrue(values["decode_triggered"])
        self.assertTrue(values["late_short_triggered"])
        self.assertEqual(values["video_dispatch_wait_image_ids"], ["i0", "i1", "i2"])
        self.assertEqual(values["video_finish_wait_image_ids"], ["i1", "i2"])
        self.assertEqual(values["dispatch_reason_counts"]["block_preempt"], 4)
        self.assertNotIn("preemptions", values)
        self.assertEqual(values["image_wait_intersections"][-1]["video_dispatch_wait_ms"], 0)
        self.assertEqual(values["image_wait_intersections"][1]["video_finish_dispatch_wait_ms"], 14.)
        self.assertIn("not GPU kernel", contention.trigger_summary(result)["scope"])

    def test_late_short_threshold_is_remaining_service_not_lateness(self):
        result = self.observed_result()
        phase = result["config"]["phases"][0]
        phase["options"] = dict(late_short_ms=10, demote_late=True)
        # These images have not missed their deadline at dispatch, but the
        # profile predicts a miss; both have observed video wait exposure.
        rows = result["phases"][0]["cells"][0]["requests"]
        rows[2]["deadline_ms"], rows[3]["deadline_ms"] = 70., 70.
        values = contention.trigger_summary(result)["arms"]["step"]
        self.assertTrue(values["late_short_triggered"])
        self.assertNotIn("i1", values["late_start_image_ids"])
        self.assertIn("i1", values["late_short_eligible_image_ids"])
        self.assertNotIn("i3", values["late_short_video_exposed_image_ids"])
        phase["options"]["late_short_ms"] = 9.
        self.assertFalse(contention.trigger_summary(result)["arms"]["step"]["late_short_triggered"])
        phase["options"].update(late_short_ms=2000, demote_late=False)
        self.assertFalse(contention.trigger_summary(result)["arms"]["step"]["late_short_triggered"])

    def test_missing_timelines_cannot_be_reported_as_observed_triggers(self):
        result = self.observed_result()
        del result["phases"][0]["cells"][0]["requests"][0]["dispatch_timeline"]
        values = contention.trigger_summary(result)["arms"]["step"]
        self.assertFalse(values["timeline_available"])
        self.assertFalse(values["pressure_triggered"])
        self.assertFalse(values["decode_triggered"])
        self.assertFalse(values["late_short_triggered"])
        self.assertIn("missing", values["unavailable_reason"])
        self.assertGreater(values["image_wait_p95_ms"], 0)

    def test_ablation_is_conditional_and_changes_one_variable_on_original_trace(self):
        with patch.object(contention, "contention_summary", return_value=dict(qualified=False)):
            self.assertEqual(contention.contention_ablation_jobs(self.study, self.profile), [])
        jobs = contention.contention_jobs(self.study, self.profile, self.capacities)
        rows = []
        for job in jobs:
            path = self.result(job, dict(status="completed"), 500.)
            rows.append(dict(job=job["id"], result_binding=api.binding(path),
                triggers=dict(arms=dict(block300=dict(late_short_triggered=True, decode_triggered=False)))))
        summary = dict(qualified=True, selected_quantum_ms=300, jobs=rows)
        with patch.object(contention, "contention_summary", return_value=summary):
            result = contention.contention_ablation_jobs(self.study, self.profile)
            self.assertEqual(contention.contention_ablation_jobs(self.study, self.profile), result)
        self.assertEqual(len(result), 1)
        job = result[0]
        self.assertEqual(job["family"], "confirmation")
        self.assertEqual(job["requires"], [j["id"] for j in jobs])
        self.assertEqual(job["metadata"]["comparison"], "contention-ablation")
        phases = api.read(job["input"])["phases"]
        self.assertEqual(phases[0]["workload"], api.read(jobs[0]["input"])["phases"][0]["workload"])
        self.assertEqual(phases[0]["workload"], phases[1]["workload"])
        self.assertEqual(dict(phases[0]["options"], late_short_ms=0), phases[1]["options"])
        self.assertTrue(all(p["record_dispatch_timeline"] for p in phases))
        self.assertFalse(job["metadata"]["independent_confirmation"])
        self.assertEqual([api.binding(b["path"]) for b in job["input_bindings"]], job["input_bindings"])


if __name__ == "__main__":
    unittest.main()
