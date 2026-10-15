import importlib.util
import json
import os
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from ditserve.research.parallel import ParallelContractError
from ditserve.research.sp_backend import PinnedWanUlyssesBackend, shard_patch_output, shard_rope_output
from ditserve.research.sp_experiment import (CASES, _launch, _validate_ready, numerical_error,
                                           rank_case, run_static_sp, static_launch_plan,
                                           _elastic_ready, run_elastic_sp, observed_path_costs,
                                           _cached_unit_budget, _comparisons)
from ditserve.research.common import digest, file_digest


class StaticLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "model_index.json").write_text("{}")
        self.config = dict(shapes={"correctness": dict(height=256, width=256, frames=17, steps=32)},
                           prompts=[dict(text="first prompt"), dict(text="second prompt")], dev_seeds=[101, 102])
        (self.root / "protocol.json").write_text(json.dumps(dict(config=self.config, sha256="protocol", source_sha256="source")))
        self.job = dict(output=str(self.root), model_path=str(model),
                        physical_devices=[dict(uuid="GPU-%d" % rank, name="identical") for rank in range(4)],
                        deadline_monotonic=time.monotonic() + 60)
        environment = patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-0,GPU-1,GPU-2,GPU-3")
        environment.start()
        self.addCleanup(environment.stop)

    def test_fixed_launch_inputs_and_distinct_requests(self):
        plan = static_launch_plan(self.job)
        self.assertEqual(plan["world_size"], 4)
        self.assertEqual(plan["local_devices"], [0, 1, 2, 3])
        self.assertEqual(plan["backend"], "nccl")
        self.assertEqual(plan["timeout_seconds"], 15)
        self.assertEqual(plan["cpu_threads_per_rank"], 8)
        self.assertEqual(plan["deadline_monotonic"], self.job["deadline_monotonic"])
        self.assertEqual(plan["cases"], list(CASES))
        self.assertTrue(all(not spec["cache"] and spec["steps"] == 32 for spec in plan["specs"]))
        self.assertNotEqual(plan["specs"][0]["prompt"], plan["specs"][1]["prompt"])
        self.assertEqual(rank_case("two_sp2", 0), ((0, 1), 0))
        self.assertEqual(rank_case("two_sp2", 3), ((2, 3), 1))
        self.assertEqual(rank_case("sp4", 3), ((0, 1, 2, 3), 0))

    def test_misassigned_devices_cache_and_shape_are_rejected(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-1,GPU-0,GPU-2,GPU-3"):
            with self.assertRaises(ValueError):
                static_launch_plan(self.job)
        for spec in (dict(cache=True), dict(frames=33), dict(steps=16), dict(threshold=.04)):
            with self.assertRaises(ValueError):
                static_launch_plan(dict(self.job, spec=spec))

    def test_prepared_state_consensus_is_request_group_scoped(self):
        payloads = {0: "A", 1: "A", 2: "B", 3: "B"}
        _validate_ready(3, payloads)  # two SP2 groups
        with self.assertRaises(ParallelContractError):
            _validate_ready(5, payloads)  # one SP4 group
        payloads[1] = "wrong latent"
        with self.assertRaises(ParallelContractError):
            _validate_ready(3, payloads)

    def _context(self, messages, *, alive=False):
        context = Mock()
        message_queue = Mock()
        pending = queue.Queue()
        for message in messages:
            pending.put(message)
        message_queue.get.side_effect = pending.get
        context.Queue.return_value = message_queue
        events = [Mock() for _ in range(7)]
        context.Event.side_effect = events
        processes = [Mock(exitcode=0) for _ in range(4)]
        for process in processes:
            process.is_alive.side_effect = [True, False] if alive else [False, False]
        context.Process.side_effect = processes
        return context, processes, events

    def test_supervisor_constructs_four_bounded_spawn_children(self):
        messages = [dict(kind="ready", rank=rank, phase=phase, payload={})
                    for phase in range(7) for rank in range(4)]
        messages += [dict(kind="complete", rank=rank, records=[]) for rank in range(4)]
        context, processes, events = self._context(messages)
        target = Mock()
        plan = static_launch_plan(self.job)
        result = _launch(plan, context=context, target=target)
        self.assertEqual(set(result), {0, 1, 2, 3})
        for rank, call in enumerate(context.Process.call_args_list):
            self.assertIs(call.kwargs["target"], target)
            self.assertEqual(call.kwargs["args"][0], rank)
            self.assertEqual(call.kwargs["args"][1], plan)
            self.assertTrue(call.kwargs["args"][2].endswith("static-sp/rendezvous"))
            processes[rank].start.assert_called_once()
        for event in events:
            event.set.assert_called_once()

    def test_failure_terminates_every_rank_without_waiting_for_collectives(self):
        context, processes, _ = self._context(
            [dict(kind="error", rank=2, error="failed step", traceback="trace")], alive=True)
        with self.assertRaisesRegex(RuntimeError, "rank 2 failed"):
            _launch(static_launch_plan(self.job), context=context, target=Mock())
        for process in processes:
            process.terminate.assert_called_once()

    def test_launcher_error_leaves_failed_artifact_not_a_pass(self):
        with patch("ditserve.research.sp_experiment._launch", side_effect=RuntimeError("fake failure")):
            result = run_static_sp(self.job)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["elastic_qualified"])
        self.assertFalse(result["serving_cost_eligible"])
        self.assertTrue((self.root / "static-sp-validation.json").is_file())
        with self.assertRaises(ValueError):
            run_static_sp(self.job)

    def test_elastic_requires_parent_and_preserves_inactive_contract(self):
        with self.assertRaisesRegex(ValueError, "charged"):
            run_elastic_sp(self.job)
        _elastic_ready(3, {0: "A", 1: "A", 2: None, 3: None})
        with self.assertRaises(ParallelContractError):
            _elastic_ready(3, {0: "A", 1: "A", 2: "stale state", 3: None})
        with patch("ditserve.research.sp_experiment._launch", side_effect=RuntimeError("fake migration failure")):
            result = run_elastic_sp(dict(self.job, parent_static_sha256="charged-static-parent"))
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["elastic_qualified"])
        self.assertFalse(result["cached_sp_qualified"])
        self.assertEqual(result["boundaries_completed_steps"], [16, 24])
        self.assertTrue((self.root / "elastic-sp-validation.json").is_file())

    def test_static_parent_attempts_elastic_only_if_complete_unit_fits(self):
        with patch("ditserve.research.sp_experiment._launch", return_value={}), \
             patch("ditserve.research.sp_experiment._comparisons", return_value=[]), \
             patch("ditserve.research.sp_experiment.run_elastic_sp") as elastic:
            result = run_static_sp(dict(self.job, deadline_monotonic=time.monotonic() + 5))
        elastic.assert_not_called()
        self.assertEqual(result["elastic"]["status"], "budget_not_fit")
        self.assertEqual(result["cached_static"]["status"], "budget_not_fit")
        self.assertEqual(result["status"], "measured_unqualified")

    def _baseline(self, seconds=80):
        value = dict(status="measured_unqualified", cache_enabled=False,
                     plan=static_launch_plan(self.job), rank_records={},
                     static_unit_wall_seconds=seconds, wall_seconds=seconds)
        value["sha256"] = digest(value)
        (self.root / "static-sp-off-unit.json").write_text(json.dumps(value))
        return value

    def test_cached_recipe_is_fixed_and_cannot_enter_elastic(self):
        plan = static_launch_plan(dict(self.job, static_recipe="t008"))
        self.assertEqual(plan["artifact_subdir"], "static-sp-t008")
        self.assertTrue(all(spec["cache"] and spec["threshold"] == .08 and spec["max_continuous"] == 1
                            for spec in plan["specs"]))
        for spec in (dict(cache=False), dict(threshold=.04), dict(max_continuous=3)):
            with self.assertRaises(ValueError):
                static_launch_plan(dict(self.job, static_recipe="t008", spec=spec))
        with self.assertRaisesRegex(ValueError, "cache-off"):
            run_elastic_sp(dict(self.job, static_recipe="t008", parent_static_sha256="parent"))

    def test_cached_unit_requires_matching_measured_baseline_and_full_budget(self):
        cached = dict(self.job, static_recipe="t008")
        with self.assertRaisesRegex(ValueError, "completed cache-off"):
            _cached_unit_budget(static_launch_plan(cached))
        self._baseline()
        budget = _cached_unit_budget(static_launch_plan(cached))
        self.assertEqual(budget["status"], "budget_not_fit")
        self.assertEqual(budget["next_complete_unit_estimate_seconds"], 100.)
        with patch("ditserve.research.sp_experiment._launch") as launch:
            result = run_static_sp(cached)
        launch.assert_not_called()
        self.assertEqual(result["status"], "budget_not_fit")
        self.assertFalse((self.root / "static-sp-t008").exists())
        self.assertTrue((self.root / "static-sp-t008-validation.json").is_file())
        changed = self._baseline()
        changed["plan"]["specs"][0]["seed"] += 1
        changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
        (self.root / "static-sp-off-unit.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            _cached_unit_budget(static_launch_plan(cached))

    def test_default_orders_complete_off_then_cached_then_cache_off_elastic(self):
        clock, calls = [100.], []
        def launch(plan):
            calls.append(plan["static_recipe"])
            self.assertEqual(plan["specs"][0]["cache"], plan["static_recipe"] == "t008")
            clock[0] += 80 if plan["static_recipe"] == "off" else 40
            return {}
        def elastic(job):
            calls.append("elastic")
            self.assertNotIn("static_recipe", job)
            return dict(status="measured_unqualified", cache_enabled=False)
        with patch("ditserve.research.sp_experiment.time.monotonic", side_effect=lambda: clock[0]), \
             patch("ditserve.research.sp_experiment._launch", side_effect=launch), \
             patch("ditserve.research.sp_experiment._comparisons", return_value=[]), \
             patch("ditserve.research.sp_experiment.run_elastic_sp", side_effect=elastic):
            result = run_static_sp(dict(self.job, deadline_monotonic=1000.,
                                       spec=dict(cache=False, threshold=0., max_continuous=1)))
        self.assertEqual(calls, ["off", "t008", "elastic"])
        self.assertEqual(result["cached_static"]["status"], "measured_unqualified")
        self.assertEqual(result["cached_static"]["complete_unit_budget"]["next_complete_unit_estimate_seconds"], 100.)
        self.assertEqual(result["static_unit_wall_seconds"], 80.)
        self.assertEqual(result["wall_seconds"], 120.)
        self.assertFalse(result["cached_static"]["cached_sp_qualified"])
        self.assertTrue((self.root / "static-sp-off-unit.json").is_file())

    def test_cached_child_failure_preserves_off_evidence_and_blocks_elastic(self):
        def launch(plan):
            if plan["static_recipe"] == "t008":
                raise RuntimeError("cached child failed")
            return {}
        with patch("ditserve.research.sp_experiment._launch", side_effect=launch), \
             patch("ditserve.research.sp_experiment._comparisons", return_value=[]), \
             patch("ditserve.research.sp_experiment.run_elastic_sp") as elastic:
            result = run_static_sp(dict(self.job, deadline_monotonic=time.monotonic() + 1000))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["static_unit_complete"])
        self.assertEqual(result["cached_static"]["status"], "failed")
        self.assertEqual(result["elastic"]["status"], "blocked_cached_static_failed")
        self.assertEqual(json.loads((self.root / "static-sp-off-unit.json").read_text())["status"], "measured_unqualified")
        elastic.assert_not_called()

    def test_cached_work_consumes_budget_before_elastic_admission(self):
        clock = [100.]
        def launch(plan):
            clock[0] += 80
            return {}
        with patch("ditserve.research.sp_experiment.time.monotonic", side_effect=lambda: clock[0]), \
             patch("ditserve.research.sp_experiment._launch", side_effect=launch), \
             patch("ditserve.research.sp_experiment._comparisons", return_value=[]), \
             patch("ditserve.research.sp_experiment.run_elastic_sp") as elastic:
            result = run_static_sp(dict(self.job, deadline_monotonic=320.))
        self.assertEqual(result["cached_static"]["status"], "measured_unqualified")
        self.assertEqual(result["elastic"]["status"], "budget_not_fit")
        elastic.assert_not_called()

    def test_observed_paths_leave_missing_reuse_cost_unmeasured(self):
        events = [dict(step_index=i, completed_steps=i + 1, path="compute", reused_forwards=0,
                       ms=10., gated_step_ms=11., capture_inclusive_ms=12.) for i in range(3)]
        result = observed_path_costs(events, expected_steps=3, guidance_scale=5.)
        self.assertFalse(result["reuse_observed"])
        self.assertIn("reuse", result["missing_observations"])
        self.assertIsNone(result["paths"]["reuse"]["ms_total"])
        events[1].update(path="mixed", reused_forwards=1)
        events[2].update(path="reuse", reused_forwards=2)
        result = observed_path_costs(events, expected_steps=3, guidance_scale=5.)
        self.assertEqual(result["observed_reused_forwards"], 3)
        self.assertEqual(result["paths"]["mixed"]["observed_steps"], 1)
        self.assertEqual(result["paths"]["reuse"]["ms_total"], 10.)
        self.assertFalse(result["cached_sp_qualified"])
        events[2]["reused_forwards"] = 0
        with self.assertRaisesRegex(ParallelContractError, "label differs"):
            observed_path_costs(events, expected_steps=3, guidance_scale=5.)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable")
    def test_cached_comparisons_keep_both_layout_and_cache_off_references(self):
        import numpy as np
        def manifests(recipe, final):
            result = {rank: [] for rank in range(4)}
            for case in CASES:
                for rank in range(4):
                    ranks, input_index = rank_case(case, rank)
                    prefix = self.root / (recipe + case + str(rank))
                    latent_path = prefix.with_suffix(".latents.npy")
                    np.save(latent_path, np.array([[0.], [final]], dtype=np.float32))
                    latent = dict(path=latent_path.name, sha256=file_digest(latent_path))
                    output = None
                    if rank == ranks[0]:
                        video_path = prefix.with_suffix(".video.npy")
                        np.save(video_path, np.array([final], dtype=np.float32))
                        output = dict(path=video_path.name, sha256=file_digest(video_path))
                    record = dict(case=case, rank=rank, input_index=input_index, spec=dict(steps=1),
                                  initial=dict(input=input_index), latents=latent, output=output,
                                  path_observations=dict(reuse_observed=False))
                    path = prefix.with_suffix(".json")
                    path.write_text(json.dumps(record))
                    result[rank].append(dict(path=path.name, sha256=file_digest(path)))
            return result
        off, cached = manifests("off", 1.), manifests("t008", 1.25)
        rows = _comparisons(self.root, cached, off_manifests=off)
        row = next(row for row in rows if row["case"] == "sp4" and row["rank"] == 0)
        self.assertEqual(row["initial_and_per_step_latent_errors"][-1]["max_abs"], 0.)
        self.assertEqual(row["cache_off_reference"]["initial_and_per_step_latent_errors"][-1]["max_abs"], .25)
        self.assertEqual(row["cache_off_reference"]["video_error"]["max_abs"], .25)
        self.assertFalse(row["cache_off_reference"]["quality_qualified"])
        self.assertFalse(row["cache_off_reference"]["lpips_measured"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable")
    def test_numeric_errors_do_not_create_a_calibrated_tolerance(self):
        import numpy as np
        result = numerical_error(np.array([1., 2.]), np.array([1., 2.1]))
        self.assertAlmostEqual(result["max_abs"], .1)
        self.assertIsNone(result["calibrated_tolerance"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["bitwise_equal"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class WanHookMappingTests(unittest.TestCase):
    def test_patch_flatten_mapping_and_sp1_bypass(self):
        import torch
        value = torch.arange(2 * 3 * 5 * 4 * 4).reshape(2, 3, 5, 4, 4)
        self.assertIs(shard_patch_output(value, 0, 1), value)
        for degree in (2, 4):
            shards = [shard_patch_output(value, rank, degree).flatten(2).transpose(1, 2)
                      for rank in range(degree)]
            self.assertTrue(torch.equal(torch.cat(shards, dim=1), value.flatten(2).transpose(1, 2)))

    def test_rope_uses_the_identical_token_intervals(self):
        import torch
        value = (torch.arange(80 * 8).reshape(1, 80, 1, 8),
                 torch.arange(80 * 8).reshape(1, 80, 1, 8) + 1000)
        self.assertIs(shard_rope_output(value, 0, 1), value)
        pieces = [shard_rope_output(value, rank, 4) for rank in range(4)]
        for index in (0, 1):
            self.assertTrue(torch.equal(torch.cat([part[index] for part in pieces], dim=1), value[index]))

    def test_unknown_layout_and_shapes_fail_closed(self):
        import torch
        with self.assertRaises(ParallelContractError):
            shard_patch_output(torch.zeros(1, 2, 3, 3, 3), 0, 4)
        with self.assertRaises(ParallelContractError):
            shard_rope_output((torch.zeros(1, 7, 1, 8),) * 2, 0, 4)
        with self.assertRaises(ParallelContractError):
            shard_patch_output(torch.zeros(1, 2, 4, 4, 4), 0, 3)

    def test_capabilities_do_not_claim_unmeasured_elasticity(self):
        self.assertFalse(PinnedWanUlyssesBackend.capabilities["gpu_validated"])
        self.assertFalse(PinnedWanUlyssesBackend.capabilities["distributed_request_transport"])
        self.assertFalse(PinnedWanUlyssesBackend.capabilities["elastic_model_equivalence_validated"])


if __name__ == "__main__":
    unittest.main()
