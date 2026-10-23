"""CPU fail-closed contracts for paired scratch performance evidence."""

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_sp2_performance as workflow


def fixture():
    config = workflow.normalize_config({})
    result = dict(schema=workflow.SCHEMA, status="completed", configuration=config, ranks=[])
    compare = dict(shape_match=True, finite=True, reference_finite=True, max_abs=0., exact=True)
    ordinary = dict(passed=True, initial="initial", steps=[f"s{i}" for i in range(20)],
                    timesteps=list(range(20)), video="video")
    for rank in (0, 1):
        data = dict(rank=rank, status="completed", backend_closed=True, ordinary_reference_before_distributed=True,
                    process_lifecycle=dict(exitcode=0, reaped=True), references=[], trials=[])
        result["ranks"].append(data)
        for seed in config["seeds"]:
            for frames in config["frames"]:
                data["references"].append(dict(seed=seed, frames=frames, ordinary=copy.deepcopy(ordinary),
                                               cross_card_sp1_exact=True))
        matrix = [(workflow.POLICIES[1], "resize_pause", 0, seed, frames)
                  for seed in config["seeds"] for frames in config["frames"]]
        matrix += [(policy, "warmup", i, seed, frames) for i, seed in enumerate(config["seeds"])
                   for frames in config["frames"] for policy in
                   (workflow.POLICIES if i % 2 == 0 else workflow.POLICIES[::-1])]
        matrix += [(policy, "measured", i, config["seeds"][0], frames) for frames in config["frames"]
                   for i in range(3) for policy in (workflow.POLICIES if i % 2 == 0 else workflow.POLICIES[::-1])]
        for policy, mode, index, seed, frames in matrix:
            key = f"{policy}-{mode}-{index}-{seed}-{frames}"
            trial = dict(state_id=key, policy=policy, mode=mode, index=index, seed=seed, frames=frames,
                         passed=True, initial_and_schedule_exact=True, initial="initial", timesteps=list(range(20)),
                         steps=[], transitions=[], final_latent="s19", video="video",
                         final_replica_audit=dict(passed=True),
                         release={key: True for key in ("states", "continuations", "ownership", "scratch", "geometry", "pending")})
            data["trials"].append(trial)
            for step in range(20):
                group = [0, 1] if mode != "resize_pause" or 5 <= step < 10 else [0]
                epoch = 1 if mode != "resize_pause" else 0 if step < 5 else 1 if step < 10 else 2
                owner = rank in group
                row = dict(step=step, group=group, epoch=epoch, owner=owner, passed=True, execution=[],
                           comparison=compare.copy(), latent=f"s{step}", replica_audit=dict(passed=True))
                trial["steps"].append(row)
                chunks = [23, 14, 23] if mode == "resize_pause" else [60]
                if owner:
                    for j, blocks in enumerate(chunks):
                        seq = step*len(chunks)+j
                        calls = (422 if blocks == 60 else 99 if blocks == 14 else 161 if j == 0 else 162) if len(group) == 2 else 0
                        scratch = policy == workflow.POLICIES[1] and len(group) == 2
                        repair = dict(policy=policy, degree=len(group), request_id=key, step_index=step, quantum_seq=seq,
                                      global_sequence=((frames-1)//4+1)*30*52,
                                      local_sequence=((frames-1)//4+1)*30*52//len(group),
                                      applied_calls=calls, unchanged_sp1_calls=60 if len(group) == 1 else 0,
                                      scratch_hits=calls-1 if scratch else 0, scratch_allocations=1 if scratch else 0,
                                      scratch_fallbacks=0, scratch_peak_bytes=1024 if scratch else 0)
                        response = dict(events=[dict(kind="block_quantum", step_index=step, step_completed=j == len(chunks)-1,
                                                    blocks=blocks, linear_repair=repair)],
                                        progress=dict(step=step+int(j == len(chunks)-1), group=group,
                                                      ownership_epoch=epoch, quantum_seq=seq+1))
                        row["execution"].append(dict(response=response, audit=dict(passed=True)))
            for j in range(2 if mode == "resize_pause" else 1):
                source, target, epoch = ([0], [0, 1], 0) if j == 0 else ([0, 1], [0], 1)
                trial["transitions"].append(dict(passed=True, checks={k: True for k in
                    ("committed", "identity", "ownership", "ready", "digest", "contract", "state", "continuation", "record", "progress")},
                    completed_steps=(5 if j == 0 else 10) if mode == "resize_pause" else 0,
                    request=dict(state_id=key, source_group=source, target_group=target, expected_epoch=epoch, transfer_id=str(j)),
                    response=dict(phase="finalized", committed=True, group=target, ownership_epoch=epoch+1,
                        transfer_ready=True, state_id=key, transfer_id=str(j),
                        execution_contract=dict(schema="ditserve.wan-sp-execution.v1", sp_linear_policy=policy))))
            if mode == "measured":
                wall = (90 if policy == workflow.POLICIES[1] else 100) + rank
                trial.update(wall_ms=wall, dit_ms=wall-10, timing_scope=workflow.TIMING_SCOPE, diagnostic_in_timing=False,
                             started_monotonic_ns=1000000000, dit_end_monotonic_ns=1000000000+(wall-10)*1000000,
                             finished_monotonic_ns=1000000000+wall*1000000)
            if mode == "warmup" and index == 0:
                values = {k: 0. for k in ("padding_host_submit_ms", "padding_cuda_ms", "linear_cuda_ms", "crop_cuda_ms")}
                trial["repair_profile"] = dict(calls=422, rows=[values.copy() for _ in range(422)], totals=values.copy())
    return result


class PairedRepairTests(unittest.TestCase):
    def test_profile_totals_reproduce_python310_left_fold_on_all_hosts(self):
        # Python 3.10 builtin sum produced this total. Python 3.12 builtin sum
        # yields 42.2, so a fixture of zeros would miss the report regression.
        values = [.1] * 422
        self.assertEqual(workflow._ordered_sum(values), 42.20000000000033)
        profiler = workflow._RepairProfile(None)
        event = types.SimpleNamespace(elapsed_time=lambda unused: .1)
        profiler.records = [dict(module="linear", padding_host_submit_ms=.1, events=[event]*4) for _ in range(422)]
        self.assertTrue(all(value == 42.20000000000033 for value in profiler.finish()["totals"].values()))
        result = fixture()
        profile = result["ranks"][0]["trials"][4]["repair_profile"]
        for row in profile["rows"]:
            row["padding_host_submit_ms"] = .1
        profile["totals"]["padding_host_submit_ms"] = 42.20000000000033
        self.assertTrue(workflow.validate_result(result)["passed"])
        profile["totals"]["padding_host_submit_ms"] = 42.2
        self.assertFalse(workflow.validate_result(result)["passed"])

    def test_actual_trial_uses_shared_transfer_path_and_normal_has_no_per_step_d2h(self):
        import numpy as np
        from test_workflow_elastic_validation import CandidateTests, Tensor
        for mode in ("resize_pause", "measured"):
            helper = CandidateTests()
            backend, advances, transitions, released = helper.backend(policy=workflow.POLICIES[1])
            repair = types.SimpleNamespace(profiler=None, scratch=None, geometry=None, pending={})
            backend.parallel = types.SimpleNamespace(handle=types.SimpleNamespace(rank=0),
                                                     linear_policy=workflow.POLICIES[1], _linear_repair=repair)
            backend.runtime._sync = lambda: None
            original = backend.advance
            def advance(request):
                response = original(request)
                audit = response["events"][0]["linear_repair"]
                blocks, seq = request["blocks"], request["quantum_seq"]
                calls = (422 if blocks == 60 else 99 if blocks == 14 else 161 if seq % 3 == 0 else 162) if audit["degree"] == 2 else 0
                audit.update(applied_calls=calls, scratch_hits=max(0, calls-1), scratch_allocations=int(calls > 0),
                             scratch_fallbacks=0, scratch_peak_bytes=1024 if calls else 0)
                return response
            backend.advance = advance
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                reference = helper.reference(root)
                metadata = json.loads((reference / "reference.json").read_text())
                metadata.update(steps=[workflow._hash_array(np.array([i+1], dtype=np.float32)) for i in range(20)],
                                video=workflow._hash_array(np.array([20], dtype=np.float32)))
                (reference / "reference.json").write_text(json.dumps(metadata))
                # Both diagnostic and normal completion gates must work from
                # the exact reference digest without retaining raw video.
                (reference / "video.npy").unlink()
                copies = []
                original_copy = workflow._cpu_array
                def cpu_array(tensor):
                    copies.append(tensor.value.copy())
                    return original_copy(tensor)
                with patch.object(workflow, "_decode_gpu", return_value=Tensor([20.])), \
                        patch.object(workflow, "_cpu_array", side_effect=cpu_array):
                    row = workflow._trial(backend, workflow.normalize_config({}), 0,
                        dict(request_id="test", frames=17, seed=3501), workflow.POLICIES[1], mode, 0,
                        reference, root / "rank0")
                self.assertTrue(row["passed"])
                self.assertEqual(row["video"], metadata["video"])
                self.assertFalse(list(root.rglob("*video*.npy")))
                self.assertTrue(all(Path(tx["path"]).parent == root / "transfers" for tx in transitions))
                self.assertEqual([r["quantum_seq"] for r in advances], list(range(60 if mode == "resize_pause" else 20)))
                if mode == "measured":
                    self.assertEqual(len(copies), 2)  # Initial identity and final latent, outside timing.
                    self.assertEqual(row["wall_ms"], (row["finished_monotonic_ns"]-row["started_monotonic_ns"])/1e6)
                else:
                    self.assertEqual(len(copies), 22)

    def test_fixed_matrix_and_normal_timing_scope(self):
        config = workflow.normalize_config({})
        self.assertEqual(config["warmup"], 2)
        self.assertEqual(config["seeds"], [3501, 3502])
        self.assertFalse(config["save_videos"])
        for changes in ({"warmup": 0}, {"repeats": 1}, {"seeds": [3501]}, {"threshold": .05},
                        {"frames": [17]}, {"policies": [workflow.POLICIES[1]]}, {"pause_blocks": [60]},
                        {"steps": 4}, {"height": 256}, {"devices": ["cuda:0", "cuda:0"]}):
            with self.assertRaises(ValueError):
                workflow.normalize_config(changes)
        self.assertIn("Excludes", workflow.TIMING_SCOPE)

    def test_complete_matrix_passes_and_reports_paired_max_rank_time(self):
        result = fixture()
        audit = workflow.validate_result(result)
        self.assertTrue(audit["passed"], audit)
        summary = workflow.summarize(result["ranks"])
        self.assertEqual(len(summary["pairs"]), 6)
        self.assertEqual(summary["pairs"][0]["arms"][workflow.POLICIES[0]]["wall_ms"], 101)
        self.assertAlmostEqual(summary["by_frames"][0]["median_wall_reduction_percent"], 1000/101)

    def test_mutations_cannot_hide_numeric_sequence_policy_release_or_timing_failures(self):
        pristine = fixture()
        mutations = [
            lambda r: r["ranks"].pop(),
            lambda r: r["ranks"][0]["trials"].pop(),
            lambda r: r["ranks"][0].update(ordinary_reference_before_distributed=False),
            lambda r: r["ranks"][0]["trials"][0]["steps"][5]["execution"][0]["response"]["progress"].update(quantum_seq=0),
            lambda r: r["ranks"][0]["trials"][0]["steps"][5]["execution"][0]["response"]["events"][0]["linear_repair"].update(policy="native"),
            lambda r: r["ranks"][0]["trials"][0]["steps"][5]["execution"][0]["response"]["events"][0]["linear_repair"].update(scratch_peak_bytes=workflow.SCRATCH_LIMIT_BYTES+1),
            lambda r: r["ranks"][0]["trials"][0]["steps"][5]["comparison"].update(max_abs=.001, exact=False),
            lambda r: r["ranks"][0]["trials"][0]["release"].update(scratch=False),
            lambda r: r["ranks"][0]["trials"][-1].update(diagnostic_in_timing=True),
            lambda r: r["ranks"][0]["trials"][-1].update(wall_ms=float("nan")),
            lambda r: r["ranks"][0]["trials"][-1].update(video="different"),
            lambda r: r["ranks"][0]["trials"][-1].update(started_monotonic_ns=1),
            lambda r: r["ranks"][0]["trials"][4]["repair_profile"]["totals"].update(linear_cuda_ms=17),
            lambda r: r["ranks"][0]["trials"][0]["transitions"][0]["response"].update(group=[0]),
            lambda r: r["ranks"][0]["trials"][0].update(initial="corrupted"),
            lambda r: r.update(summary={}),
            lambda r: r["ranks"][1]["process_lifecycle"].update(exitcode=1),
        ]
        for mutate in mutations:
            result = copy.deepcopy(pristine)
            mutate(result)
            self.assertFalse(workflow.validate_result(result)["passed"])

    def test_failure_stays_durable_and_never_performance(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_run_ranks", side_effect=TimeoutError("allowance")):
            result = workflow.run_study({}, temp)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["performance_evidence"])
            self.assertFalse(result["qualification"]["passed"])
            with self.assertRaises(FileExistsError):
                workflow.run_study({}, temp)

    def test_success_is_revalidated_before_summary_and_performance_flag(self):
        pristine = fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_run_ranks", return_value=pristine["ranks"]):
            result = workflow.run_study({}, temp)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["performance_evidence"])
        pristine["ranks"][0]["trials"][0]["passed"] = False
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_run_ranks", return_value=pristine["ranks"]):
            result = workflow.run_study({}, temp)
            self.assertEqual(result["status"], "qualification_failed")
            self.assertFalse(result["performance_evidence"])


if __name__ == "__main__":
    unittest.main()
