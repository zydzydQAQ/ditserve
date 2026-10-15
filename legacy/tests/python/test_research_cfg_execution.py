"""CPU fake-RPC integration for the finite policy-aware CFG execution path."""

import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ditserve.research.common import atomic_json, read_json
from ditserve.research.cfg_experiment import CfgExperiment
from ditserve.research.cfg_protocol import DEFAULT_CONFIG
from ditserve.research.cfg_runner import context_identity, supervise
from ditserve.research.dual_evidence import read_records
from ditserve.research.dual_matrix import inputs, seal
from test_research_dual_experiment import DeferredFakeWorker, FakeClock, bound_fake_quality


class PolicyFakeWorker(DeferredFakeWorker):
    def __init__(self, *args):
        super().__init__(*args)
        if self.role == "v100":
            self.backend = "efficient"
        self.fail_candidate_prepare = False

    def call(self, command, **args):
        if command == "prepare":
            selected = args.get("policy")
            if not selected or any(args["spec"][key] != selected[key]
                                   for key in ("cache", "threshold", "max_continuous")):
                raise AssertionError("prepare must bind the explicit matching policy")
            if self.fail_candidate_prepare and selected["id"] != "off":
                raise RuntimeError("candidate allocation failed")
        result = super().call(command, **args)
        if command == "advance" and args.get("reference_request_id"):
            reference = self.states[args["reference_request_id"]]
            reference["step"] = args["until"]
        return result


class CfgExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.clock, self.events = Path(self.temp.name), FakeClock(), []
        self.config = read_json(DEFAULT_CONFIG)
        self.frozen = dict(sha256="cfg-protocol", source_sha256="cfg-source", config=self.config)
        atomic_json(self.root / "protocol.json", self.frozen)
        self.devices = [dict(name="A100", uuid="GPU-a100"), dict(name="V100", uuid="GPU-v100")]
        self.job = dict(output=str(self.root), deadline_monotonic=time.monotonic() + 1000,
            physical_devices=self.devices, all_physical_devices=self.devices, phase="screen-a100",
            unit_estimate_seconds=1., remaining_gpu_seconds=16603.)
        self.experiment = CfgExperiment(self.job)
        self.experiment.journal.observe = lambda *a, **k: dict(status="exclusive_observed")
        self.experiment.workers = {role: PolicyFakeWorker(role, self.clock, self.events)
                                   for role in ("a100", "v100")}
        self.experiment.quality = bound_fake_quality
        self.item = inputs(self.config)[0]
        for target in ("ditserve.research.cfg_experiment.time.monotonic",
                       "ditserve.research.cfg_experiment.time.perf_counter"):
            timer = patch(target, self.clock)
            timer.start()
            self.addCleanup(timer.stop)
        mover = patch("ditserve.research.cfg_experiment.transfer", side_effect=self.transfer)
        mover.start()
        self.addCleanup(mover.stop)

    def transfer(self, source, target, request_id, action, migration_id, certificate=None):
        state = source.states.pop(request_id)
        if state["step"] != 16 or request_id in target.states:
            raise AssertionError("not a complete single-owner step16 transition")
        target.states[request_id] = state
        self.events.append(("transport", action, dict(request_id=request_id), self.clock.now))
        self.clock.advance(.01)
        return dict(action=action, byte_exact_before_first_compute=True)

    def test_004_is_explicit_and_unknown_recipe_rejected_before_rpc(self):
        spec = self.experiment.spec(self.item, "independent_004")
        self.assertTrue(spec["cache"])
        self.assertEqual(spec["threshold"], .04)
        self.assertEqual(spec["max_continuous"], 1)
        with self.assertRaises(ValueError):
            self.experiment.generate(self.item, "t008")
        self.assertEqual(self.events, [])
        run = self.experiment.generate(self.item, "independent_004")
        preparation = next(args for _, command, args, _ in self.events if command == "prepare")
        self.assertEqual(preparation["policy"]["id"], "independent_004")
        self.assertEqual(run["identity"]["policy_sha256"], preparation["policy"]["sha256"])
        self.assertEqual(run["identity"]["implementation"], "native")

    def test_each_order_has_new_candidate_and_reference_generation(self):
        candidate = "independent_004"
        self.experiment.paired_row(self.item, candidate, "a100", "dev", ["off", candidate])
        self.experiment.paired_row(self.item, candidate, "a100", "dev", [candidate, "off"])
        rows = read_records(self.root, self.frozen)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["order"] for r in rows], [["off", candidate], [candidate, "off"]])
        generations = [r[key]["generation_id"] for r in rows for key in ("candidate", "reference")]
        self.assertEqual(len(set(generations)), 4)
        self.assertTrue(all(r["fresh_pair"] and r["performance_valid"] and r["timing_eligible"] for r in rows))
        prepares = [args["policy"]["id"] for _, command, args, _ in self.events if command == "prepare"]
        self.assertEqual(prepares, ["off", candidate, candidate, "off"])

    def test_single_card_context_matches_parent_and_phase_ids_do_not_collide(self):
        expected = context_identity(self.frozen, self.devices)
        self.experiment.workers.pop("a100")
        self.experiment.job["physical_devices"] = [self.devices[1]]
        self.experiment.job["phase"] = "calibrate-v100"
        self.assertEqual(self.experiment.context(), expected)
        run = self.experiment.generate(self.item, "independent_004", role="v100")
        self.assertEqual(run["identity"]["path"], ["v100"])
        self.assertEqual(run["identity"]["attention_path"], ["efficient"])
        first_id = run["generation_id"]
        self.experiment.counter = 0
        self.experiment.job["phase"] = "holdout-v100"
        second = self.experiment.spec(self.item, "independent_004")
        self.assertNotEqual(first_id, second["request_id"])

    def test_raw_flush_occurs_after_every_decode_and_does_not_enter_timing(self):
        before = self.clock.now
        run = self.experiment.generate(self.item, "independent_004", role="v100", queue=2)
        commands = [command for _, command, _, _ in self.events]
        self.assertLess(commands.index("batch-result"), commands.index("flush"))
        self.assertLess(commands.index("finish"), commands.index("flush"))
        flush_at = [now for _, command, _, now in self.events if command == "flush"]
        latest_decode = max(r["decoded_at_monotonic"] for r in [run] + run["queued_outputs"])
        self.assertTrue(all(t >= latest_decode for t in flush_at))
        self.assertGreater((self.clock.now - before) * 1000 - run["elapsed_ms"], 1499.)
        self.assertLess(run["critical_path_ms"], 1000.)
        self.assertEqual(len(run["queued_outputs"]), 2)
        self.assertTrue(run["all_requests_decoded"])

    def test_queue_outputs_have_distinct_off_policy_and_bound_identity(self):
        run = self.experiment.generate(self.item, "independent_004", role="v100",
                                       action="full_migrate", queue=2)
        self.assertEqual(run["identity"]["path"], ["v100", "a100"])
        self.assertEqual(run["identity"]["attention_path"], ["efficient", "flash"])
        self.assertGreaterEqual(run["target_wait_ms"], 199.9)
        self.assertGreater(run["critical_path_ms"], run["target_wait_ms"])
        identifiers = [r["generation_id"] for r in [run] + run["queued_outputs"]]
        self.assertEqual(len(set(identifiers)), 3)
        for queued in run["queued_outputs"]:
            self.assertEqual(queued["policy"]["id"], "off")
            self.assertEqual(queued["identity"]["path"], ["a100"])
            self.assertEqual(queued["identity"]["implementation"], "native")
            self.assertEqual(queued["identity"]["policy_sha256"], queued["policy"]["sha256"])
        self.assertTrue(all(not w.states and not w.pending for w in self.experiment.workers.values()))

    def test_diagnostic_reference_is_separate_and_released_after_completion(self):
        run = self.experiment.generate(self.item, "independent_008", diagnostic=True)
        preparations = [args for _, command, args, _ in self.events if command == "prepare"]
        self.assertEqual([p["policy"]["id"] for p in preparations], ["off", "independent_008"])
        reference_id = preparations[0]["spec"]["request_id"]
        self.assertNotEqual(reference_id, run["generation_id"])
        advances = [args for _, command, args, _ in self.events if command == "advance"]
        self.assertTrue(all(a["reference_request_id"] == reference_id and a["diagnostic"] for a in advances))
        releases = [args["request_id"] for _, command, args, _ in self.events if command == "release"]
        self.assertEqual(releases, [reference_id])
        self.assertFalse(self.experiment.workers["a100"].states)

    def test_diagnostic_reference_is_released_if_candidate_prepare_fails(self):
        self.experiment.workers["a100"].fail_candidate_prepare = True
        with self.assertRaisesRegex(RuntimeError, "candidate allocation failed"):
            self.experiment.generate(self.item, "independent_008", diagnostic=True)
        self.assertFalse(self.experiment.workers["a100"].states)

    def test_refinement_correctness_shadow_avoids_off_trajectory_but_keeps_diagnostics(self):
        self.experiment.frozen["implementation_refinement"] = {"bound": True}
        self.experiment.job["phase"] = "correctness-a100"
        self.experiment.implementations["paired_008"] = "split_prefix"
        run = self.experiment.generate(self.item, "paired_008", instrument=True,
                                       diagnostic=True, diagnostic_reference=False)
        prepares = [args for _, command, args, _ in self.events if command == "prepare"]
        self.assertEqual([p["policy"]["id"] for p in prepares], ["paired_008"])
        advances = [args for _, command, args, _ in self.events if command == "advance"]
        self.assertTrue(all(a["diagnostic"] and a["reference_request_id"] is None for a in advances))
        self.assertEqual(run["diagnostic_scope"], "same_input_prediction_only_correctness")

    def test_no_reference_shadow_refuses_screening_and_noninstrumented_runs(self):
        for refinement, phase, instrument in ((False, "correctness-a100", True),
                (True, "screen-a100", True), (True, "correctness-a100", False)):
            with self.subTest(refinement=refinement, phase=phase, instrument=instrument):
                self.experiment.frozen["implementation_refinement"] = {"bound": True} if refinement else None
                self.experiment.job["phase"] = phase
                with self.assertRaisesRegex(ValueError, "restricted"):
                    self.experiment.generate(self.item, "paired_008", diagnostic=True,
                                            instrument=instrument, diagnostic_reference=False)
                self.assertFalse(self.events)

    def test_known_invalid_actions_are_rejected_before_any_rpc(self):
        for kwargs in (dict(role="a100", action="full_migrate"),
                       dict(role="v100", action="full_migrate", diagnostic=True),
                       dict(role="a100", queue=1), dict(action="certified_omit")):
            with self.subTest(kwargs=kwargs):
                self.events.clear()
                with self.assertRaises(ValueError):
                    self.experiment.generate(self.item, "independent_004", **kwargs)
                self.assertEqual(self.events, [])

    def test_supervisor_requires_current_cpu_pass_before_gpu_observation_or_launch(self):
        with patch("ditserve.research.cfg_runner.gpu_occupancy") as occupancy, \
                patch("ditserve.research.cfg_runner.subprocess.Popen") as launch:
            with self.assertRaisesRegex(ValueError, "cpu-validation"):
                supervise("correctness-a100", self.frozen, self.root, self.devices)
            occupancy.assert_not_called()
            launch.assert_not_called()
            bad = seal(dict(status="passed", protocol_sha256="old-protocol", source_sha256="cfg-source"))
            atomic_json(self.root / "cpu-validation.json", bad)
            with self.assertRaisesRegex(ValueError, "cpu-validation"):
                supervise("correctness-a100", self.frozen, self.root, self.devices)
            occupancy.assert_not_called()
            launch.assert_not_called()

    def test_load_binds_research_config_and_only_selected_card(self):
        self.experiment.job["physical_devices"] = [self.devices[1]]
        self.experiment.workers = {}
        with patch("ditserve.research.cfg_experiment.ResidentWorker") as resident:
            self.experiment.load()
        resident.assert_called_once()
        args, kwargs = resident.call_args
        self.assertEqual(args[:3], ("GPU-v100", "efficient", self.config["model_path"]))
        self.assertEqual(kwargs["research_config"], self.config)
        self.assertEqual(set(self.experiment.workers), {"v100"})


if __name__ == "__main__":
    unittest.main()
