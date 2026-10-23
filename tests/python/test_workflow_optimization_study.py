"""Reference preservation and bounded candidate selection; no GPU imports."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_optimization_study as optimization
from ditserve.evaluation.workflow_optimization_study import select_tiles


def replay():
    cases = []
    for seed in (3501, 3502):
        for frames in (17, 33):
            for branch in ("cond", "uncond"):
                for operator in ("to_q", "ffn.net.0.proj", "ffn.net.2"):
                    candidates = []
                    for mode, ms in (("global_tile_1024", 2.), ("global_tile_2048", 1.)):
                        arms = []
                        for shape, rank in (("full", None), ("half", 0), ("half", 1)):
                            arms.append(dict(shape_kind=shape, rank=rank, qualified=True,
                                repeated_exact=True, comparison_to_reference=dict(
                                    max_abs=.015625, finite=True, shape_match=True),
                                comparison_to_full_same_precision=dict(exact=True),
                                microbenchmark=dict(median_ms=ms)))
                        candidates.append(dict(candidate=mode, qualified=True, arms=arms))
                    cases.append(dict(seed=seed, frames=frames, branch=branch,
                                      operator=operator, candidates=candidates))
    return dict(status="completed", reference_reproduction=dict(passed=True),
                qualification=dict(source_integrity=True), cases=cases)


class OptimizationScreenTests(unittest.TestCase):
    def test_bounded_selection_uses_all_cases_and_orders_by_operator_cost(self):
        result = select_tiles(replay())
        self.assertEqual([r["mode"] for r in result], ["global_tile_2048", "global_tile_1024"])
        self.assertEqual([r["operator_cases"] for r in result], [24, 24])

    def test_candidate_self_consistency_cannot_replace_original_reference(self):
        result = replay()
        result["cases"][-1]["candidates"][1]["arms"][0]["comparison_to_reference"]["max_abs"] = .1
        self.assertEqual([r["mode"] for r in select_tiles(result)], ["global_tile_1024"])
        result["reference_reproduction"]["passed"] = False
        self.assertEqual(select_tiles(result), [])

    def test_missing_duplicate_or_nonrepeatable_case_cannot_select(self):
        for kind in ("missing", "duplicate", "repeat", "nonfinite"):
            result = replay()
            if kind == "missing":
                result["cases"].pop()
            elif kind == "duplicate":
                result["cases"][-1] = copy.deepcopy(result["cases"][0])
            else:
                for candidate in result["cases"][0]["candidates"]:
                    if kind == "repeat":
                        candidate["arms"][1]["repeated_exact"] = False
                    else:
                        candidate["arms"][1]["microbenchmark"]["median_ms"] = float("nan")
            with self.subTest(kind=kind):
                self.assertEqual(select_tiles(result), [])

    def test_both_halves_must_match_candidate_full(self):
        result = replay()
        for candidate in result["cases"][0]["candidates"]:
            candidate["arms"][2]["comparison_to_full_same_precision"]["exact"] = False
        self.assertEqual(select_tiles(result), [])

    def test_conservative_control_requires_the_complete_exact_full_row_baseline(self):
        config = dict(seeds=[3501, 3502], frames=[17, 33], steps=20)
        exact = dict(shape_match=True, finite=True, reference_finite=True, exact=True, max_abs=0.)
        ranks = [dict(rank=rank, variants=[
            dict(seed=seed, frames=frames, mode="full_rows_linear", status="qualified",
                 prefixes=[dict(prefix=prefix, passed=True, first_nonzero_step=None,
                                scheduler_steps=20, timestep_grid=list(range(20)), initial_and_schedule_exact=True,
                                video_comparison=copy.deepcopy(exact), steps=[dict(step=step, passed=True,
                                    comparison={key: copy.deepcopy(exact) for key in
                                                ("latent_before", "latent_after", "cond", "uncond")})
                                    for step in range(prefix)]) for prefix in (1, 4, 20)])
            for seed in config["seeds"] for frames in config["frames"]]) for rank in (0, 1)]
        replay_result = replay()
        replay_result.update(schema="ditserve.workflow.linear-replay.v1", qualification=dict(
            passed=True, source_integrity=True, repeated_exact=True))
        for case in replay_result["cases"]:
            case.update(qualified=True, reference_reproduced=True, arms=[dict(precision="default",
                shape_kind=kind, rank=rank, qualified=True, repeated_exact=True,
                comparison_to_reference=copy.deepcopy(exact),
                comparison_to_full_same_precision=copy.deepcopy(exact))
                for kind, rank in (("full", None), ("half", 0), ("half", 1))])
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "baseline.json"
            source_path.write_text("{}")
            ev = dict(guarded=True, qualified=False, result=dict(ranks=ranks),
                      path=source_path, launch=dict(output_sha256=optimization.api.binding(source_path)["sha256"]))
            replay_path = Path(directory) / "replay.json"
            replay_path.write_text("{}")
            replay_ev = dict(guarded=True, qualified=True, result=replay_result, path=replay_path,
                            launch=dict(output_sha256=optimization.api.binding(replay_path)["sha256"]))
            study = dict(root=directory)
            for failure in (None, "unguarded", "missing_case", "duplicate_rank", "missing_first_nonzero",
                            "latent_difference", "hidden_latent_difference", "video_difference", "short_steps",
                            "short_scheduler", "initial_mismatch", "replay_unqualified", "replay_missing_case",
                            "replay_duplicate_case", "replay_anchor", "replay_case_anchor", "anchor_comparison",
                            "ffn1_reference", "ffn1_full", "ffn1_missing_half", "repeat", "changed_binding",
                            "relaxed_threshold"):
                observed, observed_replay = copy.deepcopy(ev), copy.deepcopy(replay_ev)
                observed_config = copy.deepcopy(config)
                prefix = observed["result"]["ranks"][1]["variants"][-1]["prefixes"][-1]
                cases = observed_replay["result"]["cases"]
                ffn1 = next(case for case in cases if case["operator"] == "ffn.net.0.proj")
                if failure == "unguarded":
                    observed["guarded"] = False
                elif failure == "missing_case":
                    observed["result"]["ranks"][1]["variants"].pop()
                elif failure == "duplicate_rank":
                    observed["result"]["ranks"][1]["rank"] = 0
                elif failure == "missing_first_nonzero":
                    prefix.pop("first_nonzero_step")
                elif failure == "latent_difference":
                    prefix["first_nonzero_step"] = 3
                elif failure == "hidden_latent_difference":
                    prefix["steps"][-1]["comparison"]["latent_after"]["max_abs"] = .01
                elif failure == "video_difference":
                    prefix["video_comparison"]["exact"] = False
                elif failure == "short_steps":
                    prefix["steps"].pop()
                elif failure == "short_scheduler":
                    prefix["scheduler_steps"] = 4
                elif failure == "initial_mismatch":
                    prefix["initial_and_schedule_exact"] = False
                elif failure == "replay_unqualified":
                    observed_replay["qualified"] = False
                elif failure == "replay_missing_case":
                    cases.pop()
                elif failure == "replay_duplicate_case":
                    cases[-1] = copy.deepcopy(cases[0])
                elif failure == "replay_anchor":
                    observed_replay["result"]["reference_reproduction"]["passed"] = False
                elif failure == "replay_case_anchor":
                    cases[0]["reference_reproduced"] = False
                elif failure == "anchor_comparison":
                    cases[0]["arms"][0]["comparison_to_reference"]["max_abs"] = .01
                elif failure == "ffn1_reference":
                    ffn1["arms"][1]["comparison_to_reference"]["exact"] = False
                elif failure == "ffn1_full":
                    ffn1["arms"][2]["comparison_to_full_same_precision"]["max_abs"] = .01
                elif failure == "ffn1_missing_half":
                    ffn1["arms"].pop()
                elif failure == "repeat":
                    ffn1["arms"][2]["repeated_exact"] = False
                elif failure == "changed_binding":
                    observed_replay["launch"]["output_sha256"] = "changed"
                elif failure == "relaxed_threshold":
                    observed_config["threshold"] = .1
                with self.subTest(failure=failure), \
                        patch.object(optimization, "_original_sp", return_value=observed_config), \
                        patch.object(optimization, "current_job", side_effect=lambda s, name: dict(id=name)), \
                        patch.object(optimization, "evidence", side_effect=lambda s, job:
                                     observed if job["id"] == "sp-linear-controls" else observed_replay), \
                        patch.object(optimization.api, "save"), \
                        patch.object(optimization.api, "add_job",
                                     return_value=dict(id="sp-conservative-control", input_bindings=[])) as add:
                    if failure:
                        with self.assertRaises(ValueError):
                            optimization.conservative_job(study)
                        add.assert_not_called()
                    else:
                        job = optimization.conservative_job(study)
                        self.assertEqual(add.call_args.args[3]["variants"], ["full_rows_except_ffn1"])
                        self.assertEqual(add.call_args.args[3]["seeds"], config["seeds"])
                        self.assertFalse(add.call_args.kwargs["metadata"]["performance_evidence"])
                        self.assertEqual([item["path"] for item in job["input_bindings"]],
                                         [str(source_path), str(replay_path)])
                        self.assertEqual(add.call_args.kwargs["metadata"]["replay_output_sha256"],
                                         replay_ev["launch"]["output_sha256"])

    def test_warmed_confirmation_cannot_hide_live_growth_or_missing_plateau(self):
        original = dict(frame_pairs=[[17, 17]], lifecycle_pairs=3,
                        steady_reserved_growth_bytes=64 * 1024**2)
        arms = []
        for index, name in enumerate(("serial", "streams-current-decode", "streams-shared-decode")):
            arms.append(dict(id=name, frame_pair=[17, 17],
                launch=dict(pid=100+index, reaped=True, returncode=0, timed_out=False,
                            started_unix_ns=index*10, finished_unix_ns=index*10+9),
                result=dict(completed_pairs=3, process=dict(pid=100+index),
                qualification=dict(passed=name == "streams-shared-decode", scopes={
                    scope: dict(status="passed") for scope in ("numeric", "capacity", "lifecycle")}),
                steady_state=dict(metrics=dict(
                    allocated_bytes=dict(maximum_growth_bytes=0),
                    reserved_bytes=dict(values=[10, 20, 20]),
                    nvml_used_bytes=dict(values=[11, 21, 21]))))))
        ev = dict(guarded=True, result=dict(arms=arms, configuration=original,
            qualification=dict(scopes=dict(numeric=dict(status="passed")), control_cross_arm=[
                dict(id="streams-current-decode", frames_a=17, frames_b=17, observed_pairs=3,
                     expected_pairs=3, complete=True, exact=True)])),
            path=Path("/fixture/result.json"), launch=dict(output_sha256="bound-result"))
        source = dict(id="overlap-isolated-lifecycle", input="/fixture/input.json",
                      configuration_sha256=optimization.api.digest(original))
        with tempfile.TemporaryDirectory() as directory:
            for failure in (None, "live_growth", "no_plateau", "cross_arm_mismatch",
                            "control_missing", "control_partial", "control_mismatch", "control_count",
                            "control_pair", "reused_pid", "wrong_pid", "not_reaped", "timeout",
                            "returncode", "overlapping_children", "missing_launch_time"):
                observed = copy.deepcopy(ev)
                serial = observed["result"]["arms"][0]["result"]["steady_state"]["metrics"]
                if failure == "live_growth":
                    serial["allocated_bytes"]["maximum_growth_bytes"] = 1
                elif failure == "no_plateau":
                    serial["reserved_bytes"]["values"][-1] = 21
                elif failure == "cross_arm_mismatch":
                    observed["result"]["qualification"]["scopes"]["numeric"]["status"] = "failed"
                elif failure == "control_missing":
                    observed["result"]["qualification"]["control_cross_arm"] = []
                elif failure and failure.startswith("control_"):
                    field, value = {"control_partial": ("complete", False),
                                    "control_mismatch": ("exact", False),
                                    "control_count": ("observed_pairs", 2),
                                    "control_pair": ("frames_b", 33)}[failure]
                    observed["result"]["qualification"]["control_cross_arm"][0][field] = value
                elif failure:
                    middle = observed["result"]["arms"][1]
                    if failure == "reused_pid":
                        middle["launch"]["pid"] = middle["result"]["process"]["pid"] = 100
                    elif failure == "wrong_pid":
                        middle["result"]["process"]["pid"] = 999
                    elif failure == "missing_launch_time":
                        middle["launch"].pop("finished_unix_ns")
                    else:
                        field, value = {"not_reaped": ("reaped", False), "timeout": ("timed_out", True),
                                        "returncode": ("returncode", -9),
                                        "overlapping_children": ("started_unix_ns", 8)}[failure]
                        middle["launch"][field] = value
                with self.subTest(failure=failure), \
                        patch.object(optimization, "current_job", return_value=source), \
                        patch.object(optimization, "evidence", return_value=observed), \
                        patch.object(optimization.api, "read", return_value=copy.deepcopy(original)), \
                        patch.object(optimization.api, "binding", return_value=dict(path=str(ev["path"]))), \
                        patch.object(optimization.api, "save"), \
                        patch.object(optimization.api, "add_job", return_value=dict(
                            id="overlap-isolated-warmed-lifecycle", input_bindings=[])) as add:
                    if failure:
                        with self.assertRaises(ValueError):
                            optimization.warmed_overlap_job(dict(root=directory))
                        add.assert_not_called()
                    else:
                        optimization.warmed_overlap_job(dict(root=directory))
                        config = add.call_args.args[3]
                        self.assertEqual((config["warmup_pairs"], config["lifecycle_pairs"]), (2, 3))
                        self.assertEqual(config["steady_reserved_growth_bytes"],
                                         original["steady_reserved_growth_bytes"])
                        self.assertFalse(add.call_args.kwargs["metadata"]["performance_evidence"])


if __name__ == "__main__":
    unittest.main()
