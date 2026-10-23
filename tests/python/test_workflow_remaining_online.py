"""Deterministic CPU-only matrices and immutable configuration contracts."""

import copy
from pathlib import Path
import unittest

from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation import workflow_remaining_online as study


def inputs():
    profile=api.read(Path(__file__).resolve().parents[2]/"configs/profiles/rtx4090-sp1-local.json")
    for frames in (17,33):
        profile["classes"][f"wan1.3b-480x832x{frames}"]["steps"]=20
    return dict(jobs=[],selections=dict(cpu=dict(cores=8,depth=1))),profile,dict(centers_ms=[2000.,4000.,6000.,8000.],mixed_capacity=.4)


class RemainingOnlineStudyTests(unittest.TestCase):
    def test_twenty_jobs_are_pure_deterministic_and_dependency_ordered(self):
        args=inputs();before=copy.deepcopy(args)
        jobs=study.templates(*args)
        self.assertEqual(len(jobs),20)
        self.assertEqual(args,before)
        self.assertEqual(jobs,study.templates(*args))
        self.assertEqual([j["metadata"]["seed"] for j in jobs[:2]],[4600,4700])
        seen=set(study.BASE_PREREQUISITES)
        for job in jobs:
            self.assertTrue(set(job["requires"])<=seen)
            seen.add(job["id"])
            self.assertTrue(study.validate_config(job["config"],job["metadata"]))
            self.assertGreaterEqual(job["timeout_seconds"],3600)
            self.assertGreater(job["timeout_seconds"],job["estimated_seconds"])
            self.assertFalse(job["metadata"]["performance_blocks_confirmation"])
        self.assertEqual(sum(len(j["config"]["phases"]) for j in jobs),80)

    def test_four_arm_controls_and_ab_ba_ab_quantum_pairs(self):
        jobs=study.templates(*inputs())
        for family in ("quantum_repeated","quantum_unique","poisson","high_load"):
            rows=[j for j in jobs if j["metadata"]["family"]==family and j["metadata"]["stage"]=="confirmation"]
            for repeat,job in enumerate(rows):
                phases=job["config"]["phases"]
                self.assertEqual([p["id"] for p in phases],[["off300","off200","on300","on200"],
                    ["on200","on300","off200","off300"],["on300","on200","off300","off200"]][repeat])
                for phase in phases:
                    self.assertEqual(phase["occupancy_backend"],"nvml")
                    self.assertFalse(phase["dispatch_refresh"])
                    self.assertEqual(phase["controller_cpu_affinity"],[59])
                    self.assertEqual(phase["text_prefetch_depth"],1)
                    for slot in (0,1):
                        overrides=phase["worker_overrides"]
                        self.assertEqual(overrides[f"wan{slot}"]["cpu_threads"],8)
                        self.assertEqual(overrides[f"sana{slot}"]["cpu_threads"],4)
                        self.assertEqual(bool(overrides[f"sana{slot}"]["options"]),phase["id"].startswith("on"))

    def test_unique_excludes_warmup_and_poisson_has_40_requests(self):
        for job in study.templates(*inputs()):
            phase=job["config"]["phases"][0]
            images=[r for r in phase["workload"] if r["kind"]=="image"]
            videos=[r for r in phase["workload"] if r["kind"]=="video"]
            large=job["metadata"]["family"] in ("poisson","high_load")
            self.assertEqual((len(images),len(videos)),(32,8) if large else (20,4))
            if job["metadata"]["family"]=="quantum_unique":
                prompts={r["spec"]["prompt"] for r in images}
                self.assertEqual(len(prompts),20)
                self.assertFalse(prompts&{r["spec"]["prompt"] for r in phase["warmup"]})
            for frames in (17,33):
                negatives=[r["spec"]["negative_prompt"] for r in videos if r["spec"]["frames"]==frames]
                self.assertEqual(negatives.count(""),len(negatives)//2)

    def test_high_load_is_explicit_compression_with_unchanged_slo_slack(self):
        _,profile,anchors=inputs()
        expected=api.trajectory(profile,5101,count=40,pattern="poisson",load=.8,lambda0=.4)
        actual=study.workload(profile,anchors,"high_load",5101)
        for left,right in zip(expected,actual):
            self.assertEqual(right["arrival_ms"],left["arrival_ms"]/1.25)
            self.assertAlmostEqual(right["deadline_ms"]-right["arrival_ms"],left["deadline_ms"]-left["arrival_ms"])
        with self.assertRaisesRegex(ValueError,"capacities.mixed"):
            study.templates({},profile,dict(centers_ms=anchors["centers_ms"]))

    def test_scheduler_and_decode_have_exactly_the_preregistered_interventions(self):
        jobs=study.templates(*inputs())
        scheduler=next(j for j in jobs if j["metadata"]["family"]=="scheduler")
        phases={p["id"]:p for p in scheduler["config"]["phases"]}
        self.assertEqual(phases["S0"]["policy"],"static_sp1")
        for name in ("S2","S4"):self.assertEqual(phases[name]["options"]["late_short_ms"],0)
        for name in ("S3","S4"):
            self.assertFalse(phases[name]["options"]["chunked_decode"])
            self.assertNotIn("decode_frames",phases[name]["options"])
        decode=next(j for j in jobs if j["metadata"]["family"]=="decode")
        self.assertEqual([p["options"].get("decode_frames") for p in decode["config"]["phases"]],[1,2,None])

    def test_rehashed_control_trace_and_metadata_tampering_is_rejected(self):
        original=study.templates(*inputs())[0]
        edits=[lambda j:j["config"]["phases"][1]["options"].update(quantum_ms=100),
               lambda j:j["config"]["phases"][0].update(occupancy_backend="nvidia_smi"),
               lambda j:j["config"]["phases"][0]["workload"][0].update(arrival_ms=999.),
               lambda j:j["metadata"].update(performance_blocks_confirmation=True),
               lambda j:j["metadata"].update(rule="relaxed"),
               lambda j:j["metadata"].update(comparisons=[]),
               lambda j:j["config"]["phases"][0]["worker_overrides"]["wan0"].update(cpu_threads=16)]
        for edit in edits:
            job=copy.deepcopy(original);edit(job)
            with self.assertRaises(ValueError):study.validate_config(job["config"],job["metadata"])
        args=list(inputs());args[0]["jobs"]=[dict(id="historical",metadata=dict(seed=4601))]
        with self.assertRaisesRegex(ValueError,"already used"):study.templates(*args)


if __name__=="__main__":unittest.main()
