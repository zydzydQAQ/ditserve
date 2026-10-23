"""CPU tests for elastic GPU-harness gates, transitions and evidence scope."""

import json
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.evaluation import workflow_elastic_validation as workflow


class ConfigTests(unittest.TestCase):
    def test_fixed_qualification_matrix_cannot_relax_accuracy_or_change_schedule(self):
        value = workflow.normalize_config({})
        self.assertEqual(value["frames"], [17, 33])
        self.assertEqual(len(value["seeds"]), 2)
        self.assertEqual(value["group_timeout_seconds"], 60)
        for changes in ({"steps": 4}, {"threshold": .1}, {"group_timeout_seconds": 120},
                        {"schedule": [{"step": 4, "degree": 2}]}, {"pause_blocks": [60]},
                        {"pause_blocks": [23, 36]}, {"seeds": [1, 1]},
                        {"sp_linear_policy": "full_rows_ffn2"}):
            with self.assertRaises(ValueError):
                workflow.normalize_config(changes)

    def test_first_failure_blocks_every_dependent_gate(self):
        self.assertEqual(workflow.gate_order([]), "sp1_resume")
        self.assertEqual(workflow.gate_order([dict(gate="sp1_resume", passed=True)]), "sp1_transfer")
        self.assertIsNone(workflow.gate_order([dict(gate="sp1_resume", passed=False)]))
        self.assertIsNone(workflow.gate_order([dict(gate="fixed_sp2", passed=True)]))

    def test_each_stage_selects_only_its_required_reference_and_gate(self):
        expected = dict(transfer=("sp1_resume", "sp1_transfer"), pause=("fixed_sp2", "fixed_sp2_pause"),
                        resize=("elastic_1_2_1",), interleave=("fixed_sp2", "sp2_sana_interleave"))
        for stage, gates in expected.items():
            config = workflow.normalize_config(dict(stage=stage))
            self.assertEqual(workflow.STAGE_GATES[config["stage"]], gates)
            passed = [dict(gate=gate, passed=True) for gate in gates]
            self.assertIsNone(workflow.gate_order(passed, gates))
        with self.assertRaises(ValueError):
            workflow.normalize_config(dict(stage="unsupported"))

    def test_failure_and_partial_ranks_are_durable_and_never_performance(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(workflow, "_run_ranks", side_effect=TimeoutError("allowance")):
            result = workflow.run_validation({}, temp)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["qualification"]["passed"])
            self.assertFalse(result["performance_evidence"])
            self.assertIn("allowance", result["error"])
            self.assertEqual(json.loads((Path(temp)/"run.json").read_text())["status"], "failed")
            with self.assertRaises(FileExistsError):
                workflow.run_validation({}, temp)

    def test_empty_or_single_rank_cannot_qualify(self):
        for ranks in ([], [dict(status="completed", qualification=dict(passed=True))]):
            with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_run_ranks", return_value=ranks):
                self.assertEqual(workflow.run_validation({}, temp)["status"], "qualification_failed")

    def test_shared_video_references_are_removed_after_completion_and_timeout(self):
        for fail in (False, True):
            video_directories = []
            def run_ranks(config, out, video_root):
                self.assertFalse(config["save_videos"])
                directory = Path(video_root)
                video_directories.append(directory)
                (directory / "video.npy").write_bytes(b"temporary comparison video")
                (out / "step-00.npy").write_bytes(b"retained latent evidence")
                if fail:
                    raise TimeoutError("both children terminated and reaped")
                return [dict(rank=rank, status="completed", qualification=dict(passed=True)) for rank in (0, 1)]
            with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_run_ranks", side_effect=run_ranks):
                result = workflow.run_validation({}, temp)
                self.assertEqual(result["status"], "failed" if fail else "completed")
                self.assertTrue((Path(temp) / "step-00.npy").exists())
                self.assertFalse(list(Path(temp).rglob("*video*.npy")))
                self.assertTrue(all(not path.exists() for path in video_directories))

    def test_explicit_video_retention_does_not_use_temporary_references(self):
        def run_ranks(config, out, video_root):
            self.assertTrue(config["save_videos"])
            self.assertIsNone(video_root)
            (out / "video.npy").write_bytes(b"explicit retained output")
            return []
        with tempfile.TemporaryDirectory() as temp, patch.object(workflow, "_run_ranks", side_effect=run_ranks):
            workflow.run_validation(dict(save_videos=True), temp)
            self.assertTrue((Path(temp) / "video.npy").exists())

    def test_fast_rank_exit_failure_cannot_be_hidden_by_completed_manifests(self):
        process = types.SimpleNamespace(start=lambda: None, is_alive=lambda: False,
                                        join=lambda **kwargs: None, exitcode=7)
        context = types.SimpleNamespace(Process=lambda **kwargs: process)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(workflow.multiprocessing, "get_context", return_value=context):
            root = Path(temp)
            for rank in (0, 1):
                (root / f"rank{rank}").mkdir()
                (root / f"rank{rank}" / "run.json").write_text(json.dumps(
                    dict(status="completed", qualification=dict(passed=True))))
            with self.assertRaisesRegex(RuntimeError, "exited 7 during finalization"):
                workflow._run_ranks(workflow.normalize_config({}), root)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    float = cpu = detach

    def numpy(self):
        return self.value

    def tolist(self):
        return self.value.tolist()

    def item(self):
        return self.value.item()


class OrdinaryReferenceTests(unittest.TestCase):
    def test_default_reference_saves_latents_and_video_hash_but_no_video_array(self):
        for retain in (False, True):
            state = types.SimpleNamespace(latents=Tensor([0.]), step_index=0,
                                          scheduler=types.SimpleNamespace(timesteps=Tensor(np.arange(20))))
            released = []
            def step(value):
                value.step_index += 1
                value.latents = Tensor([value.step_index])
            runtime = types.SimpleNamespace(prepare=lambda spec: state, step=step,
                finish=lambda value: value.latents.numpy(), release=lambda value: released.append(value))
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                kwargs = dict(save_video=True) if retain else {}
                result = workflow._ordinary_reference(runtime, dict(request_id="reference"), out, save=True, **kwargs)
                self.assertTrue(result["passed"])
                self.assertEqual(result["video"], workflow._hash_array(np.array([20.], dtype=np.float32)))
                self.assertEqual(len(list(out.glob("step-*.npy"))), 20)
                self.assertEqual((out / "video.npy").exists(), retain)
                self.assertEqual(json.loads((out / "reference.json").read_text())["video"], result["video"])
            self.assertEqual(released, [state])


class CandidateTests(unittest.TestCase):
    def backend(self, fail_step=None, policy="native"):
        states, advances, transitions, released = {}, [], [], []
        continuations, records, parked = {}, {}, {}
        def prepare(request):
            frames = request.get("spec", {}).get("frames", 17)
            states[request["state_id"]] = types.SimpleNamespace(
                step_index=0, partial=0, latents=Tensor([0.]),
                global_sequence=((frames-1)//4+1)*30*52,
                scheduler=types.SimpleNamespace(timesteps=Tensor(np.arange(20))))
            continuations[request["state_id"]] = object()
            records[request["state_id"]] = dict(group=[0], epoch=0)
        def advance(request):
            advances.append(request)
            state = states[request["state_id"]]
            before_step = state.step_index
            state.partial += request["blocks"]
            if state.partial == 60:
                state.step_index += 1
                state.partial = 0
                offset = .1 if state.step_index == fail_step else 0.
                state.latents = Tensor([state.step_index + offset])
            record = records[request["state_id"]]
            return dict(progress=dict(step=state.step_index, next_block=state.partial,
                                      group=record["group"], ownership_epoch=record["epoch"],
                                      quantum_seq=request["quantum_seq"]+1),
                        events=[dict(kind="block_quantum", step_index=before_step,
                                     blocks=request["blocks"], step_completed=state.partial == 0,
                                     linear_repair=dict(policy=policy, degree=len(record["group"]),
                                         request_id=request["state_id"], step_index=before_step,
                                         quantum_seq=request["quantum_seq"],
                                         global_sequence=state.global_sequence,
                                         local_sequence=state.global_sequence//len(record["group"]),
                                         unchanged_sp1_calls=120 if len(record["group"]) == 1 else 0,
                                         applied_calls=120 if len(record["group"]) == 2 else 0))])
        def release(request):
            released.append(request["state_id"])
            del states[request["state_id"]]
            continuations.pop(request["state_id"])
            records.pop(request["state_id"])
        def distributed(request):
            transitions.append(request)
            key, target = request["state_id"], request["target_group"]
            if 0 not in target:
                parked[key] = states.pop(key)
                continuations.pop(key)
                records.pop(key)
            else:
                if key in parked:
                    states[key] = parked.pop(key)
                    continuations[key] = object()
                records[key] = dict(group=target, epoch=request["expected_epoch"]+1)
            progress = (dict(step=states[key].step_index, ownership_epoch=request["expected_epoch"]+1,
                             group=target, transfer_ready=True) if 0 in target else None)
            return dict(phase="finalized", committed=True, transfer_id=request["transfer_id"], state_id=key,
                        ownership_epoch=request["expected_epoch"]+1, group=target, state_sha256="a"*64,
                        transfer_ready=True, progress=progress, execution_contract=dict(
                            schema="ditserve.wan-sp-execution.v1", sp_linear_policy=policy))
        def all_gather_object(peers, local):
            key = next(iter(states))
            other_owner = 1 in records[key]["group"] or key.endswith("sp1_resume")
            peers[:] = [local, dict(rank=1, owner=other_owner, latent=local["latent"] if other_owner else None)]
        torch = types.SimpleNamespace(
            int32="int32", tensor=lambda value, **_: Tensor(value),
            distributed=types.SimpleNamespace(
                barrier=lambda: None, all_reduce=lambda *args, **kwargs: None,
                all_gather_object=all_gather_object,
                ReduceOp=types.SimpleNamespace(MIN="min")))
        runtime = types.SimpleNamespace(torch=torch, device="cuda:0",
                                        finish=lambda state: state.latents.numpy())
        backend = types.SimpleNamespace(states=states, prepare=prepare, advance=advance, release=release,
                                        runtime=runtime, distributed=distributed, continuations=continuations,
                                        elastic=types.SimpleNamespace(records=records))
        return backend, advances, transitions, released

    def reference(self, root):
        path = root / "rank0" / "case" / "ordinary"
        path.mkdir(parents=True)
        for step in range(20):
            np.save(path / f"step-{step:02d}.npy", np.array([step+1], dtype=np.float32))
        np.save(path / "video.npy", np.array([20], dtype=np.float32))
        (path / "reference.json").write_text(json.dumps(dict(
            initial=workflow._hash_array(np.array([0.], dtype=np.float32)),
            timesteps=list(range(20)))))
        return path

    def test_elastic_switches_only_after_steps_five_and_ten_with_global_sequence(self):
        backend, advances, transitions, released = self.backend()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.reference(root)
            out = root / "rank0" / "case" / "elastic_1_2_1"
            result = workflow._candidate(backend, workflow.normalize_config({}), 0,
                                         dict(request_id="case"), "elastic_1_2_1", reference, out)
            self.assertTrue(result["passed"])
            self.assertEqual([row["source_group"] for row in transitions], [[0], [0, 1]])
            self.assertEqual([row["target_group"] for row in transitions], [[0, 1], [0]])
            self.assertEqual([row["expected_epoch"] for row in transitions], [0, 1])
            self.assertTrue(all(Path(row["path"]).parent == root / "transfers" for row in transitions))
            self.assertEqual([row["quantum_seq"] for row in advances], list(range(20)))
            self.assertEqual([row["expected_epoch"] for row in advances], [0]*5+[1]*5+[2]*10)
            self.assertFalse(result["performance_evidence"])
        self.assertFalse(backend.states)
        self.assertEqual(len(released), 1)

    def test_pause_resumes_every_step_exactly_and_releases_state(self):
        backend, advances, _, released = self.backend()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = workflow._candidate(backend, workflow.normalize_config({}), 0,
                                         dict(request_id="case"), "sp1_resume", self.reference(root),
                                         root / "rank0" / "case" / "sp1_resume")
            self.assertTrue(result["passed"])
            self.assertEqual([row["blocks"] for row in advances], [23, 37]*20)
            self.assertEqual([row["quantum_seq"] for row in advances], list(range(40)))
        self.assertEqual(len(released), 1)

    def test_pause_after_both_conditional_and_unconditional_blocks_preserves_global_sequence(self):
        backend, advances, _, released = self.backend()
        config = workflow.normalize_config(dict(pause_blocks=[23, 14, 23]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = workflow._candidate(backend, config, 0, dict(request_id="case"),
                                         "sp1_resume", self.reference(root),
                                         root / "rank0" / "case" / "sp1_resume")
            self.assertTrue(result["passed"])
            self.assertEqual([row["blocks"] for row in advances], [23, 14, 23]*20)
            self.assertEqual([row["quantum_seq"] for row in advances], list(range(60)))
            self.assertEqual([quantum["next_block"] for quantum in result["steps"][0]["quanta"]],
                             [23, 37, 0])
        self.assertEqual(len(released), 1)

    def test_numerical_failure_stops_before_transition_or_later_steps(self):
        backend, advances, transitions, released = self.backend(fail_step=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "rank0" / "case" / "elastic_1_2_1"
            result = workflow._candidate(backend, workflow.normalize_config({}), 0,
                                         dict(request_id="case"), "elastic_1_2_1", self.reference(root), out)
            self.assertFalse(result["passed"])
            self.assertEqual(result["first_failure"], dict(stage="latent", step=1))
            self.assertEqual(len(advances), 2)
            self.assertFalse(transitions)
            self.assertTrue((out / "failed-step-01.npy").is_file())
            self.assertEqual(result["not_run_steps"], list(range(2, 20)))
        self.assertEqual(len(released), 1)

    def test_interleave_executes_sana_requests_between_real_wan_quanta(self):
        backend, advances, transitions, released = self.backend()
        images, states = [], {}
        def prepare(request):
            images.append(request)
            states[request["state_id"]] = dict(output=np.array([7.], dtype=np.float32))
        sana = types.SimpleNamespace(
            states=states, prepare=prepare, advance=lambda _: None,
            finish=lambda _: dict(output=dict(sha256="image")),
            release=lambda request: states.pop(request["state_id"]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = workflow._candidate(
                backend, workflow.normalize_config(dict(stage="interleave", pause_blocks=[23, 14, 23])), 0,
                dict(request_id="case", seed=123), "sp2_sana_interleave", self.reference(root),
                root / "rank0" / "case" / "sp2_sana_interleave", sana=sana,
                sana_reference=workflow._hash_array(np.array([7.], dtype=np.float32)))
            self.assertTrue(result["passed"])
            self.assertEqual(len(images), 4)
            self.assertEqual([row["step"] for row in result["sana_interruptions"]], [0, 5, 10, 15])
            self.assertEqual([row["after_blocks"] for row in result["sana_interruptions"]], [23]*4)
            self.assertEqual([row["blocks"] for row in advances], [23, 14, 23]*20)
            self.assertTrue(all(row["passed"] for row in result["sana_interruptions"]))
        self.assertFalse(states)
        self.assertFalse(backend.states)

    def test_interleave_missing_real_image_backend_cannot_pass(self):
        backend, _, _, _ = self.backend()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "real Sana"):
                workflow._candidate(
                    backend, workflow.normalize_config(dict(stage="interleave")), 0,
                    dict(request_id="case", seed=123), "sp2_sana_interleave", self.reference(root),
                    root / "rank0" / "case" / "sp2_sana_interleave")
        self.assertFalse(backend.states)

    def test_default_video_failure_retains_error_and_hash_without_saving_video(self):
        for retain in (False, True):
            backend, _, _, _ = self.backend()
            backend.runtime.finish = lambda state: np.array([20.1], dtype=np.float32)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                out = root / "rank0" / "case" / "elastic_1_2_1"
                result = workflow._candidate(backend, workflow.normalize_config(dict(save_videos=retain)), 0,
                    dict(request_id="case"), "elastic_1_2_1", self.reference(root), out)
                self.assertFalse(result["passed"])
                self.assertEqual(result["first_failure"], dict(stage="video"))
                self.assertGreater(result["video_comparison"]["max_abs"], .05)
                self.assertTrue(result["video"]["finite"])
                self.assertEqual((out / "failed-video.npy").exists(), retain)

    def test_fixed_sp2_pause_uses_temporary_full_arrays_and_keeps_exact_gate(self):
        backend, _, _, _ = self.backend()
        config = workflow.normalize_config(dict(stage="pause"))
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as videos:
            root = Path(temp)
            reference = self.reference(root)
            video_path = workflow._video_reference_path(reference, videos)
            video_path.parent.mkdir(parents=True)
            (reference / "video.npy").rename(video_path)
            fixed = root / "rank0" / "case" / "fixed_sp2"
            result = workflow._candidate(backend, config, 0, dict(request_id="case"), "fixed_sp2",
                reference, fixed, video_root=videos)
            self.assertTrue(result["passed"])
            self.assertTrue(workflow._video_reference_path(fixed, videos).is_file())
            self.assertFalse((fixed / "video.npy").exists())
            backend.runtime.finish = lambda state: np.array([20.001], dtype=np.float32)
            result = workflow._candidate(backend, config, 0, dict(request_id="case"), "fixed_sp2_pause",
                fixed, root / "rank0" / "case" / "fixed_sp2_pause", video_root=videos)
            self.assertFalse(result["passed"])
            self.assertGreater(result["video_comparison"]["max_abs"], 0)
            self.assertLess(result["video_comparison"]["max_abs"], .05)
            self.assertFalse(list(root.rglob("*video*.npy")))

    def test_repaired_elastic_retains_real_block_policy_transition_and_release_evidence(self):
        backend, advances, _, _ = self.backend(policy="full_rows_except_ffn1")
        config = workflow.normalize_config(dict(sp_linear_policy="full_rows_except_ffn1", stage="resize"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = workflow._candidate(backend, config, 0, dict(request_id="case"),
                "elastic_1_2_1", self.reference(root), root / "rank0" / "case" / "elastic_1_2_1")
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["release_audit"].values()))
        self.assertEqual([row["completed_steps"] for row in result["transitions"]], [5, 10])
        self.assertTrue(all(row["passed"] for row in result["transitions"]))
        repairs = [row["execution"][0]["events"][0]["linear_repair"] for row in result["steps"]]
        self.assertEqual([row["degree"] for row in repairs], [1]*5+[2]*5+[1]*10)
        self.assertEqual([row["applied_calls"] for row in repairs], [0]*5+[120]*5+[0]*10)
        self.assertTrue(all(row["replica_audit"]["passed"] for row in result["steps"]))
        self.assertEqual(len(advances), 20)

    def test_exact_outputs_do_not_qualify_a_backend_that_ignored_the_repair_policy(self):
        backend, advances, _, _ = self.backend(policy="native")
        config = workflow.normalize_config(dict(sp_linear_policy="full_rows_except_ffn1", stage="resize"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = workflow._candidate(backend, config, 0, dict(request_id="case"),
                "elastic_1_2_1", self.reference(root), root / "rank0" / "case" / "elastic_1_2_1")
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_failure"], dict(stage="backend_execution_or_replica", step=0))
        self.assertEqual(len(advances), 1)
        self.assertTrue(all(result["release_audit"].values()))

    def test_transition_that_keeps_a_leaving_rank_state_is_rejected(self):
        backend, _, _, _ = self.backend(policy="full_rows_except_ffn1")
        backend.prepare(dict(state_id="case"))
        original = backend.distributed
        def leak(request):
            state = backend.states["case"]
            response = original(request)
            backend.states["case"] = state
            return response
        backend.distributed = leak
        transitions = []
        with self.assertRaisesRegex(RuntimeError, "source release"):
            workflow._transition(backend, "case", [0, 1], [1], 0, Path("transfer.pt"), transitions,
                                 rank=0, completed_steps=5, audit=True)
        self.assertFalse(transitions[0]["audit"]["owner_state"])


class RepairedEvidenceTests(unittest.TestCase):
    def complete_result(self):
        helper = CandidateTests()
        policy = "full_rows_except_ffn1"
        config = workflow.normalize_config(dict(sp_linear_policy=policy, stage="pause",
                                               seeds=[3501, 3502], pause_blocks=[23, 14, 23], save_videos=True))
        rank = dict(rank=0, status="completed", qualification=dict(passed=True), cases=[],
                    ordinary_reference_before_distributed=True, backend_init=dict(sp_linear_policy=policy),
                    backend_closed=True, process_lifecycle=dict(exitcode=0, reaped=True))
        with tempfile.TemporaryDirectory() as temp:
            for seed in config["seeds"]:
                for frames in config["frames"]:
                    case_id = f"seed{seed}-f{frames}"
                    root = Path(temp) / case_id
                    reference = helper.reference(root)
                    reference_metadata = json.loads((reference / "reference.json").read_text())
                    backend, _, _, _ = helper.backend(policy=policy)
                    rows = []
                    for gate in workflow.STAGE_GATES["pause"]:
                        candidate = workflow._candidate(backend, config, 0, dict(request_id=case_id, frames=frames), gate,
                            reference, root / "rank0" / "case" / gate)
                        rows.append(candidate)
                        reference = root / "rank0" / "case" / "fixed_sp2"
                    rank["cases"].append(dict(case=case_id, seed=seed, frames=frames, passed=True,
                        cross_card_sp1_exact=True, gates=rows,
                        ordinary=dict(passed=True, steps=[{}]*20, **reference_metadata)))
        peer = copy.deepcopy(rank)
        peer["rank"] = 1
        return dict(configuration=config, ranks=[rank, peer])

    def test_complete_repaired_matrix_passes_cpu_revalidation(self):
        result = self.complete_result()
        audit = workflow.validate_repaired_result(result)
        self.assertTrue(audit["passed"], audit["errors"])
        self.assertEqual(audit["scope"], "one_stage_full_matrix_not_performance")

    def test_boolean_summaries_missing_coverage_and_forged_event_audits_cannot_pass(self):
        reference = self.complete_result()
        def event(result):
            return result["ranks"][0]["cases"][0]["gates"][0]["steps"][0]["execution"][0]
        changes = (
            lambda result: result["ranks"].pop(),
            lambda result: result["ranks"][1].update(rank=0),
            lambda result: result["ranks"][0]["cases"].pop(),
            lambda result: result["ranks"][0].update(ordinary_reference_before_distributed=False),
            lambda result: result["ranks"][0].update(backend_closed=False),
            lambda result: result["ranks"][0]["cases"][0]["gates"][0]["steps"].pop(),
            lambda result: event(result)["events"][0]["linear_repair"].update(applied_calls=0),
            lambda result: event(result)["events"][0]["linear_repair"].update(policy="native"),
            lambda result: event(result)["events"][0].update(kind="step"),
            lambda result: result["ranks"][0]["cases"][0]["gates"][0]["transitions"].clear(),
            lambda result: result["ranks"][0]["cases"][0]["gates"][0]["transitions"][0]["response"].update(
                execution_contract=dict(schema="ditserve.wan-sp-execution.v1", sp_linear_policy="native")),
            lambda result: result["ranks"][0]["cases"][0]["gates"][0]["transitions"][0]["request"].update(
                target_group=[1]),
            lambda result: result["ranks"][0]["cases"][0]["gates"][0]["initial_identity"].update(actual_initial={}),
            lambda result: result["ranks"][0]["cases"][0]["gates"][0]["release_audit"].update(states_cleared=False),
        )
        for change in changes:
            result = copy.deepcopy(reference)
            change(result)
            self.assertFalse(workflow.validate_repaired_result(result)["passed"])
        for malformed in ({}, dict(configuration=None), dict(configuration={"seeds": None}),
                          dict(configuration=reference["configuration"], ranks=[None])):
            self.assertFalse(workflow.validate_repaired_result(malformed)["passed"])


if __name__ == "__main__":
    unittest.main()
