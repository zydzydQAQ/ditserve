"""Frozen adaptive confirmation and fail-closed causal dispatch evidence."""

import copy
import math
from pathlib import Path
import tempfile
import unittest

from ditserve.evaluation import workflow_adaptive_study as adaptive
from ditserve.evaluation import workflow_contention_study as contention
from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation.workflow_budget import atomic_json


REPO = Path(__file__).resolve().parents[2]


class AdaptiveStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = dict(root=str(self.root), run_id="run03", source=str(REPO), source_digest="fixture",
            source_revisions=[], budget_seconds=28800, family_caps_seconds=api.CAPS, image="sha256:fixture",
            jobs=[], decisions=[], selections=dict(cpu=dict(cores=8, depth=1), old_selection=dict(value=7)),
            gpu_mapping=[dict(ordinal=i, uuid=f"GPU-00000000-0000-0000-0000-{i:012}", name="RTX 4090") for i in range(2)])
        self.profile = api.read(REPO / "configs/profiles/rtx4090-sp1-local.json")
        for frames, encode, step, finish in ((17, 10242., 582., 1968.), (33, 9575., 1347., 3860.)):
            self.profile["classes"][f"wan1.3b-480x832x{frames}"].update(
                steps=20, encode_ms=encode, step_compute_ms=step, finish_ms=finish)
        self.profile["classes"]["sana-1024"]["service_ms"] = 233.
        self.calibration = dict(profile=self.profile, capacities=dict(video=.07046, mixed=.3136))
        atomic_json(self.root / "profile.json", self.profile)
        atomic_json(self.root / "calibration.json", self.calibration)
        for label in ("video", "mixed"):
            job = api.add_job(self.study, "calibration-" + label, "calibration", {}, module="fixture")
            videos = [dict(request_id=f"v{i}", kind="video", status="completed", spec=dict(frames=33),
                start_ms=13610. if i == 0 else 11490., finish_ms=45340. if i == 0 else 43010.) for i in range(4)]
            images = [dict(request_id=f"i{i}", kind="image", status="completed") for i in range(16)]
            phases = [dict(id=f"repeat{i}", status="completed", cells=[dict(status="completed",
                elapsed_ms=64000. if label == "mixed" else 114000.,
                requests=copy.deepcopy(videos + (images if label == "mixed" else [])))]) for i in range(3)]
            self.result(job, dict(status="completed", phases=phases), 462. if label == "mixed" else 612.)
        for job in contention.contention_jobs(self.study, self.profile, self.calibration):
            self.result(job, dict(status="completed"), 500.)

    def result(self, job, value, wall=500.):
        path = Path(job["output"]) / "run.json"
        atomic_json(path, value)
        atomic_json(self.root / "launches" / job["id"] / "launch.json", dict(status="completed",
            output_sha256=api.binding(path)["sha256"], wall_seconds=wall,
            gpu_monitor=dict(errors=[], conflicts=[], memory={str(i): dict(samples=3,
                minimum_free_bytes=3 * 1024**3, total_bytes=24 * 1024**3) for i in range(2)})))

    def observed(self, job):
        config = api.read(job["input"])
        phases = []
        for phase in config["phases"]:
            rows = copy.deepcopy(phase["workload"])
            first = min(row["arrival_ms"] for row in rows if row["kind"] == "image")
            for ordinal, row in enumerate(rows):
                name = phase["id"]
                is_adaptive = name == "adaptive"
                hot = is_adaptive and ordinal % 2 == 1
                observed = first + 10 if hot else 0.
                last = first if hot else None
                chosen = 100 if hot or name == "block100" else 300
                feedback = dict(adaptive=is_adaptive, mode="hot" if hot else "cool" if is_adaptive else "fixed",
                    trigger="recent_image_arrival" if hot else "quiet" if is_adaptive else "fixed",
                    observed_at_ms=observed, waiting_images=0, oldest_image_wait_ms=0.,
                    last_image_arrival_ms=last, hot_window_ms=1000 if is_adaptive else 0)
                event = dict(start_ms=max(observed + 1, row["arrival_ms"]), finish_ms=max(observed + 2, row["arrival_ms"] + 1),
                    group=[ordinal % 2], reason="admit", quantum_feedback=feedback)
                events = []
                if row["kind"] == "video":
                    cost = self.profile["classes"][row["class"]]
                    blocks = max(1, min(math.floor(chosen / (cost["step_compute_ms"] / cost["blocks_per_step"])), cost["blocks_per_step"]))
                    for unused in range(20):
                        remaining = cost["blocks_per_step"]
                        while remaining:
                            executed = min(blocks, remaining)
                            events.append(dict(event, quantum=blocks, block_count=blocks, chosen_quantum_ms=chosen, executed_blocks=executed))
                            remaining -= executed
                    events += [dict(event, quantum="finish", decode_frames=1) for unused in range(cost["decode_chunks"])]
                else:
                    events.append(dict(event, quantum="atomic"))
                row.update(status="completed", start_ms=event["start_ms"], finish_ms=row["arrival_ms"] + 100000.,
                    dispatch_timeline=events, outputs=dict(video=dict(finite=True, sha256=row["request_id"]),
                                                         latent=dict(finite=True, sha256="latent-" + row["request_id"])))
            phases.append(dict(id=phase["id"], status="completed", cells=[dict(status="completed", elapsed_ms=150000., requests=rows)]))
        return dict(schema="ditserve.run.v1", status="completed", config=config, phases=phases)

    def test_freezes_three_independent_rotated_arms_and_retains_old_state(self):
        old_jobs, old_selections = copy.deepcopy(self.study["jobs"]), copy.deepcopy(self.study["selections"])
        jobs = adaptive.adaptive_jobs(self.study)
        self.assertEqual(self.study["jobs"][:len(old_jobs)], old_jobs)
        self.assertEqual(self.study["selections"], old_selections)
        orders = []
        for job, seed in zip(jobs, adaptive.SEEDS):
            self.assertEqual(job["id"], f"adaptive-quantum-s{seed}")
            self.assertEqual(job["family"], "confirmation")
            self.assertLessEqual(job["estimated_seconds"], 550)
            phases = api.read(job["input"])["phases"]
            orders.append([phase["id"] for phase in phases])
            self.assertEqual(len(phases[0]["workload"]), 24)
            self.assertTrue(all(phase["workload"] == phases[0]["workload"] for phase in phases))
            for phase in phases:
                self.assertEqual(phase["options"]["decode_frames"], 1)
                self.assertEqual(phase["options"]["late_short_ms"], 2000)
                self.assertTrue(phase["options"]["demote_late"])
                self.assertEqual(phase["text_prefetch_depth"], 1)
                self.assertTrue(phase["record_dispatch_timeline"])
            self.assertEqual([api.binding(bound["path"]) for bound in job["input_bindings"]], job["input_bindings"])
            self.assertEqual(len(job["input_bindings"]), 17)
        self.assertEqual(orders, [["block300", "block100", "adaptive"], ["block100", "adaptive", "block300"], ["adaptive", "block300", "block100"]])
        self.assertEqual(adaptive.adaptive_jobs(self.study), jobs)
        self.study["source_digest"] = "new-revision"
        with self.assertRaisesRegex(ValueError, "frozen adaptive"):
            adaptive.adaptive_jobs(self.study)

    def test_causal_records_are_complete_but_no_speedup_is_inferred(self):
        for job in adaptive.adaptive_jobs(self.study):
            self.result(job, self.observed(job))
        summary = adaptive.summarize(self.study)
        self.assertTrue(summary["qualified"], str(summary))
        self.assertFalse(summary["benefit_confirmed"])
        self.assertEqual(summary["recommendation"], "retain_fixed300")
        for job in summary["jobs"]:
            self.assertEqual(job["feedback"]["arms"]["adaptive"]["executed_blocks"], 4800)
            self.assertEqual(job["feedback"]["arms"]["adaptive"]["one_frame_decode_dispatches"], 28)

    def test_rejects_future_feedback_wrong_decode_and_missing_actual_blocks(self):
        job = adaptive.adaptive_jobs(self.study)[0]
        original = self.observed(job)
        self.assertTrue(adaptive.feedback_audit(original)["passed"])
        for mutation in ("future", "decode", "actual", "duplicate", "missing_request"):
            value = copy.deepcopy(original)
            phase = next(row for row in value["phases"] if row["id"] == "adaptive")
            rows = phase["cells"][0]["requests"]
            event = rows[0]["dispatch_timeline"][0]
            if mutation == "future":
                event["quantum_feedback"]["last_image_arrival_ms"] = 9999999.
            elif mutation == "decode":
                next(event for event in rows[0]["dispatch_timeline"] if event["quantum"] == "finish")["decode_frames"] = 2
            elif mutation == "actual":
                event.pop("executed_blocks")
            elif mutation == "duplicate":
                value["phases"].append(copy.deepcopy(phase))
            else:
                rows.pop()
            self.assertFalse(adaptive.feedback_audit(value)["passed"], mutation)

    def test_historical_bindings_and_memory_cannot_be_silently_relaxed(self):
        adaptive.adaptive_jobs(self.study)
        path = self.root / "launches/contention-s3701/launch.json"
        value = api.read(path)
        value["gpu_monitor"]["memory"]["0"]["minimum_free_bytes"] = 0
        atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, "guarded historical"):
            adaptive.adaptive_jobs(self.study)

    def test_run_saves_only_new_selection_and_does_not_depend_on_sp(self):
        before = copy.deepcopy(self.study["selections"])
        seen = []
        def run(study, job):
            seen.append(job["id"])
            self.result(job, self.observed(job))
        summary = adaptive.run_followup(self.study, run)
        self.assertEqual(len(seen), 3)
        self.assertTrue(summary["qualified"])
        self.assertEqual({key: value for key, value in self.study["selections"].items() if key != adaptive.SELECTION}, before)


if __name__ == "__main__":
    unittest.main()
