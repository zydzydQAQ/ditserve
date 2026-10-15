"""CPU fake-RPC checks of complete trajectories and experiment composition."""

import copy
import contextlib
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research.budget import BudgetStop
from ditserve.research.dual_experiment import DualExperiment, IneligibleAction
from ditserve.research.dual_matrix import fits_plan, inputs
from ditserve.research.dual_protocol import DEFAULT_CONFIG
from ditserve.research.dual_runtime import correctness_state_digest, RemoteError


QUALITY = dict(status="ok", finite=True, range_valid=True, frame_count=17,
               lpips_per_frame=[.01] * 17, lpips_mean=.01)


def bound_fake_quality(candidate, reference):
    return dict(copy.deepcopy(QUALITY), candidate_output_sha256=candidate["output_sha256"],
                reference_output_sha256=reference["output_sha256"],
                candidate_identity_sha256=digest(candidate["identity"]),
                reference_identity_sha256=digest(reference["identity"]))


class FakeClock:
    def __init__(self):
        self.now = 100.

    def advance(self, seconds):
        self.now += seconds

    def __call__(self):
        return self.now


class FakeWorker:
    def __init__(self, role, clock, events):
        self.role, self.clock, self.events = role, clock, events
        self.device, self.pending, self.states = "GPU-" + role, False, {}
        self.backend, self.eligible = ("flash" if role == "a100" else "math"), True

    def call(self, command, **args):
        if self.pending:
            raise AssertionError("RPC submitted while a real queue is running")
        self.events.append((self.role, command, copy.deepcopy(args), self.clock.now))
        self.clock.advance(.001)
        if command == "warm":
            return dict(mode="frozen_cpu_embeddings_mechanism_only")
        if command == "backend":
            if self.states:
                raise AssertionError("backend changed with live requests")
            self.backend = args["attention"]
            return self.backend
        if command == "prepare":
            spec = args["spec"]
            self.states[spec["request_id"]] = dict(spec=spec, step=0, events=[])
            return dict(request_id=spec["request_id"], text_encoding=dict(mode="frozen_cpu_embeddings_mechanism_only"))
        if command == "advance":
            state = self.states[args["request_id"]]
            steps = list(range(state["step"] + 1, args["until"] + 1))
            factor = .002 if self.role == "v100" else .001
            factor *= .7 if self.backend == "efficient" else 1.
            self.clock.advance(len(steps) * factor)
            state["step"] = args["until"]
            state["events"].extend(dict(kind="step", step_index=index - 1, path="compute") for index in steps)
            return dict(step=state["step"], events=state["events"], inventories=[dict(step=state["step"])],
                        traces=[dict(step=index, state_sha256="step-%d" % index) for index in steps] if args.get("instrument") else [])
        if command == "checkpoint":
            return dict(state_path=args["request_id"] + "-state.pkl", state_sha256="state-file")
        if command == "certificate":
            return dict(eligible=self.eligible, reasons=[] if self.eligible else ["not_a_refresh_boundary"])
        if command == "local_roundtrip":
            self.clock.advance(.005)
            return dict(action=args["action"], byte_exact_before_first_compute=True)
        if command == "finish":
            state = self.states.pop(args["request_id"])
            if state["step"] != 32:
                raise AssertionError("truncated trajectory decoded")
            self.clock.advance(.02)
            result = self.output(args["request_id"])
            result["events"] = state["events"]
            return result
        if command == "release":
            self.states.pop(args["request_id"], None)
            return None
        raise AssertionError("unsupported fake command " + command)

    def output(self, request_id):
        return dict(status="completed", output_path="arrays/" + request_id + "-video.npy",
                    output_sha256=digest("same-video"), latent_sha256="same-latent", artifact_io_ms=3.,
                    decoded_at_monotonic=self.clock.now, events=[])

    def start(self, command, **args):
        if command != "batch" or self.pending:
            raise AssertionError("only an idle worker can start the real queue")
        self.events.append((self.role, "batch-start", copy.deepcopy(args), self.clock.now))
        self.pending, self.queued = True, args["specs"]
        self.queue_ready = self.clock.now + .1 * len(self.queued)

    def result(self):
        if not self.pending:
            raise AssertionError("no real queue outstanding")
        self.clock.now = max(self.clock.now, self.queue_ready)
        self.events.append((self.role, "batch-result", {}, self.clock.now))
        self.pending = False
        return [self.output(spec["request_id"]) for spec in self.queued]


class DeferredFakeWorker(FakeWorker):
    """Archive writes are deliberately much slower than the fake GPU work."""
    def __init__(self, *args):
        super().__init__(*args)
        self.pending_artifacts = set()

    def output(self, request_id):
        result = super().output(request_id)
        result.pop("output_sha256")
        result.update(request_id=request_id, artifacts_pending=True, artifact_io_ms=0.)
        self.pending_artifacts.add(request_id)
        return result

    def call(self, command, **args):
        if command != "flush":
            if command == "finish" and not args.get("defer_artifacts"):
                raise AssertionError("mechanism finish must defer archival work")
            return super().call(command, **args)
        if self.pending:
            raise AssertionError("flush began before the real request queue finished")
        self.events.append((self.role, "flush", copy.deepcopy(args), self.clock.now))
        result = {}
        for request_id in args["request_ids"]:
            if request_id not in self.pending_artifacts:
                raise AssertionError("flush did not preserve request ownership")
            self.pending_artifacts.remove(request_id)
            self.clock.advance(.5)
            result[request_id] = dict(output_sha256=digest("same-video"), artifacts_pending=False, artifact_io_ms=500.)
        return result


class DualExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.clock, self.events = Path(self.temp.name), FakeClock(), []
        self.config = read_json(DEFAULT_CONFIG)
        self.frozen = dict(sha256="protocol", source_sha256="source", config=self.config)
        atomic_json(self.root / "protocol.json", self.frozen)
        self.devices = [dict(name="A100", uuid="GPU-a100"), dict(name="V100", uuid="GPU-v100")]
        self.job = dict(output=str(self.root), deadline_monotonic=time.monotonic() + 1000,
                        physical_devices=self.devices, all_physical_devices=self.devices, phase="migration",
                        unit_estimate_seconds=1., remaining_gpu_seconds=28800)
        self.experiment = DualExperiment(self.job)
        self.experiment.journal.observe = lambda *a, **k: dict(status="exclusive_observed")
        self.experiment.workers = {role: FakeWorker(role, self.clock, self.events) for role in ("a100", "v100")}
        timer = patch("ditserve.research.dual_experiment.time.perf_counter", self.clock)
        timer.start()
        self.addCleanup(timer.stop)
        monotonic = patch("ditserve.research.dual_experiment.time.monotonic", self.clock)
        monotonic.start()
        self.addCleanup(monotonic.stop)
        mover = patch("ditserve.research.dual_experiment.transfer", side_effect=self.transfer)
        mover.start()
        self.addCleanup(mover.stop)
        self.experiment.quality = bound_fake_quality
        self.item = inputs(self.config)[0]

    def transfer(self, source, target, request_id, action, migration_id, certificate=None):
        state = source.states.pop(request_id)
        if state["step"] != 16 or request_id in target.states:
            raise AssertionError("migration is not at the complete single-owner step16 boundary")
        target.states[request_id] = state
        self.events.append(("transport", action, dict(request_id=request_id), self.clock.now))
        self.clock.advance(dict(full_migrate=.01, cold_migrate=.006, certified_omit=.004)[action])
        return dict(action=action, byte_exact_before_first_compute=True)

    def test_full_trajectory_migrates_once_after_step16_and_records_all32_steps(self):
        row = self.experiment.generate(self.item, "t008", role="v100", action="full_migrate", instrument=True)
        self.assertEqual([trace["step"] for trace in row["traces"]], list(range(1, 33)))
        advances = [(role, args["until"]) for role, command, args, _ in self.events if command == "advance"]
        self.assertEqual(advances, [("v100", 16), ("a100", 32)])
        self.assertEqual(sum(role == "transport" for role, _, _, _ in self.events), 1)
        self.assertEqual(row["identity"]["path"], ["v100", "a100"])
        self.assertEqual(row["identity"]["boundary"], 16)
        self.assertTrue(all(not worker.states for worker in self.experiment.workers.values()))

    def test_single_v100_calibration_does_not_require_resident_a100(self):
        self.experiment.workers.pop("a100")
        self.experiment.job["phase"] = "calibrate-v100"
        row = self.experiment.generate(self.item, role="v100")
        self.assertEqual(row["identity"]["path"], ["v100"])
        self.assertTrue(row["all_requests_decoded"])

    def test_real_queue_wait_and_all_decode_completion_are_in_critical_path(self):
        row = self.experiment.generate(self.item, "t008", role="v100", action="full_migrate", queue=2)
        commands = [command for _, command, _, _ in self.events]
        self.assertLess(commands.index("batch-result"), commands.index("full_migrate"))
        self.assertGreaterEqual(row["target_wait_ms"], 200. - 1e-6)
        self.assertGreater(row["critical_path_ms"], row["target_wait_ms"])
        self.assertEqual(len(row["queued_outputs"]), 2)
        self.assertTrue(row["all_requests_decoded"])
        self.assertEqual(row["artifact_io_sum_ms"], 9.)

    def test_stay_overlaps_real_queue_and_waits_until_every_request_decodes(self):
        row = self.experiment.generate(self.item, "t008", role="v100", queue=2)
        commands = [command for _, command, _, _ in self.events]
        self.assertLess(commands.index("finish"), commands.index("batch-result"))
        self.assertEqual(row["target_wait_ms"], 0.)
        self.assertGreaterEqual(row["critical_path_ms"], 200. - 1e-6)
        self.assertFalse(self.experiment.workers["a100"].pending)

    def test_illegal_certificate_releases_source_without_migration(self):
        self.experiment.workers["v100"].eligible = False
        with self.assertRaises(IneligibleAction):
            self.experiment.generate(self.item, "t008", role="v100", action="certified_omit")
        self.assertFalse(self.experiment.workers["v100"].states)
        self.assertFalse(any(role == "transport" for role, _, _, _ in self.events))

    def test_migration_validation_binds_both_refs_and_all_dev_holdout_inputs(self):
        atomic_json(self.root / "calibrate.json", dict(cache_gates={role: dict(eligible=True) for role in ("a100", "v100")}))
        summary = self.experiment.migration()
        self.assertEqual(summary["status"], "completed")
        import json
        rows = [json.loads(line) for line in (self.root / "records.jsonl").read_text().splitlines()]
        quality_rows = [row for row in rows if row["type"] == "migration_quality"]
        self.assertEqual(len(quality_rows), 12)
        for row in quality_rows:
            common, path, candidate = (row[key]["identity"] for key in ("common_reference", "path_reference", "candidate"))
            self.assertEqual(common["input"], path["input"])
            self.assertEqual(common["input"], candidate["input"])
            self.assertEqual(common["path"], ["a100"])
            self.assertEqual(path["path"], ["v100", "a100"])
            self.assertEqual((common["recipe"], path["recipe"]), ("off", "off"))
            self.assertEqual(candidate["recipe"], "t008")
        self.assertEqual(sum(row["split"] == "dev" for row in quality_rows), 8)
        self.assertEqual(sum(row["split"] == "holdout" for row in quality_rows), 4)

    def test_each_calibration_card_runs_complete_eight_input_pair_matrix(self):
        self.experiment.workers.pop("a100")
        self.experiment.job["phase"] = "calibrate-v100"
        summary = self.experiment.calibrate()
        self.assertTrue(summary["cache_gates"]["v100"]["eligible"])
        prepared = [args["spec"] for _, command, args, _ in self.events if command == "prepare"]
        self.assertEqual(len(prepared), 16)
        self.assertEqual(sum(spec["cache"] for spec in prepared), 8)
        self.assertEqual({spec["seed"] for spec in prepared}, {101, 102})

    def test_correctness_digest_canonicalizes_only_default_set_and_never_mutates_live_metadata(self):
        config = dict(_use_default_values=["z", "a"], disable_corrector=[2, 1])
        payload = dict(scheduler=dict(_internal_dict=config))
        worker = SimpleNamespace(torch=SimpleNamespace(is_tensor=lambda _: False), _payload=lambda _: payload)
        state = SimpleNamespace(step_index=16, needs_anchor=False, current_step_refreshed=False)
        first = correctness_state_digest(worker, state)
        self.assertEqual(config["_use_default_values"], ["z", "a"])
        config["_use_default_values"] = ["a", "z"]
        self.assertEqual(first, correctness_state_digest(worker, state))
        config["disable_corrector"] = [1, 2]
        self.assertNotEqual(first, correctness_state_digest(worker, state))

    def test_probe_revalidates_both_backend_orders_after_uniform_shape_fallback(self):
        self.experiment.job["phase"] = "probe"
        with patch("ditserve.research.dual_experiment.fits_plan", side_effect=[False, True, True, True, True]), \
                patch.object(self.experiment, "correctness") as correctness:
            summary = self.experiment.probe()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["shape"], self.config["shapes"]["fallback"])
        self.assertEqual(summary["backends"]["v100"], "efficient")
        correctness.assert_called_once_with(self.item)
        import json
        rows = [json.loads(line) for line in (self.root / "records.jsonl").read_text().splitlines()]
        pairs = [row for row in rows if row["type"] == "backend_pair"]
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(row["shape"] == self.config["shapes"]["fallback"] for row in pairs))
        self.assertEqual([row["order"] for row in pairs], [["math", "efficient"], ["efficient", "math"]])

    def test_probe_falls_back_when_only_complete_correctness_phase_budget_does_not_fit(self):
        self.experiment.job["phase"] = "probe"
        self.experiment.started_monotonic = self.clock.now
        self.experiment.deadline = self.clock.now + 200.
        self.experiment.job["phase_remaining_gpu_seconds"] = dict(
            correctness=300., calibration=10000., migration=10000., mechanism=10000., capacity=0., reserve=2000.)
        original = self.experiment.generate

        def measured(*args, **kwargs):
            run = original(*args, **kwargs)
            # Both complete plans fit the total budget. Main's measured costs
            # exceed only the remaining correctness allowance; fallback fits.
            if self.experiment.shape == self.config["shapes"]["main"]:
                run["elapsed_ms"] *= 100
            return run

        with patch.object(self.experiment, "generate", side_effect=measured), \
                patch("ditserve.research.dual_experiment.fits_plan", wraps=fits_plan) as admission, \
                patch.object(self.experiment, "correctness") as correctness:
            summary = self.experiment.probe()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["shape"], self.config["shapes"]["fallback"])
        correctness.assert_called_once_with(self.item)
        self.assertEqual(admission.call_count, 5)
        for index, call in enumerate(admission.call_args_list):
            plan, remaining, phases = call.args
            self.assertEqual(plan["required_phases"].count("correctness"), 1)
            self.assertAlmostEqual(plan["required_gpu_seconds"],
                sum(plan["phases_gpu_seconds"][key] for key in plan["required_phases"]))
            self.assertAlmostEqual(plan["planned_with_optional_gpu_seconds"], sum(plan["phases_gpu_seconds"].values()))
            self.assertLess(plan["required_gpu_seconds"], remaining)
            self.assertLess(phases["correctness"], 300.)  # Already-spent probe time is deducted.
            if index == 0:
                self.assertGreater(plan["phases_gpu_seconds"]["correctness"], phases["correctness"])
                self.assertFalse(fits_plan(plan, remaining, phases))
            else:
                self.assertLessEqual(plan["phases_gpu_seconds"]["correctness"], phases["correctness"])
                self.assertTrue(fits_plan(plan, remaining, phases))

    def planning_seed(self, costs):
        return dict(validated=True, shape=self.config["shapes"]["main"], measured_costs_seconds=costs,
                    source_run="archived-run01", source_protocol_sha256="old-protocol",
                    source_sha256="old-source", source_unit_sha256="sealed-main-baselines")

    def conditional_fallback_probe(self, *, candidate_seconds=13., candidate_quality=True):
        estimate = dict(validated=True, shape=copy.deepcopy(self.config["shapes"]["fallback"]),
            backend="efficient", v100_seconds=13., source_run="archived-run02",
            source_protocol_sha256="previous-protocol", source_sha256="previous-source",
            source_units=[dict(unit_id="fallback-efficient-0", sha256="first-unit"),
                          dict(unit_id="fallback-efficient-1", sha256="second-unit")],
            physical_devices=copy.deepcopy(self.devices))
        seed = self.planning_seed(dict(a100=22., v100=174.))
        seed["fallback_backend_cost_estimate"] = estimate
        self.experiment.job.update(phase="probe", planning_seed=seed,
            phase_remaining_gpu_seconds=dict(correctness=1090., calibration=10000., migration=10000.,
                                              mechanism=10000., reserve=2000.))
        self.experiment.started_monotonic = self.clock.now
        self.experiment.deadline = self.clock.now + 2000.
        self.clock.advance(75.)  # Model/encoding setup counts on both resident devices.
        original = self.experiment.generate

        def measured(*args, **kwargs):
            role = kwargs.get("role", "a100")
            backend = self.experiment.workers[role].backend
            seconds = (candidate_seconds if backend == "efficient" else 24.7081) if role == "v100" else (
                6.3408 if backend == "flash" else 20.)
            started = self.clock.now
            row = original(*args, **kwargs)
            self.clock.advance(seconds - (self.clock.now - started))
            row["elapsed_ms"] = seconds * 1000
            return row

        def quality(candidate, reference):
            value = bound_fake_quality(candidate, reference)
            if candidate["identity"]["attention_path"] == ["efficient"]:
                self.clock.advance(50.)  # Fresh full-frame quality is also charged.
                if not candidate_quality:
                    value.update(lpips_mean=.06, lpips_per_frame=[.06] * 17)
            return value

        self.experiment.quality = quality
        with patch.object(self.experiment, "generate", side_effect=measured), \
                patch.object(self.experiment, "correctness") as correctness:
            summary = self.experiment.probe()
        return summary, correctness, estimate

    def test_conditional_cost_seed_requalifies_both_orders_and_uses_fresh_costs_for_final_plan(self):
        summary, correctness, estimate = self.conditional_fallback_probe(candidate_seconds=15.)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["backends"]["v100"], "efficient")
        correctness.assert_called_once_with(self.item)
        preliminary = read_json(self.root / "cost-plan-fallback-conditional.json")
        conditional = preliminary["conditional_backend_estimate"]
        self.assertFalse(conditional["qualification_reused"])
        self.assertTrue(conditional["requires_fresh_backend_qualification"])
        self.assertEqual(conditional["source_units"], estimate["source_units"])
        self.assertAlmostEqual(preliminary["correctness_components_gpu_seconds"]["pending_backend_pairs"],
                               4 * (2 * 24.7081 * 1.3 + 30))
        second = read_json(self.root / "cost-plan-fallback-conditional-1.json")
        self.assertEqual(second["conditional_backend_estimate"]["used_v100_seconds"], 15.)
        self.assertEqual(summary["cost_plan"]["measured_costs_seconds"]["v100"], 15.)
        self.assertNotIn("conditional_backend_estimate", summary["cost_plan"])
        import json
        pairs = [r for r in map(json.loads, (self.root / "records.jsonl").read_text().splitlines())
                 if r["type"] == "backend_pair"]
        self.assertEqual([row["order"] for row in pairs], [["math", "efficient"], ["efficient", "math"]])
        self.assertTrue(all(row["protocol_sha256"] == "protocol" for row in pairs))

    def test_advisory_candidate_never_qualifies_failed_fresh_quality_or_bypasses_math_cost_gate(self):
        summary, correctness, _ = self.conditional_fallback_probe(candidate_quality=False)
        self.assertEqual(summary["status"], "budget_stopped")
        correctness.assert_not_called()
        actual_plan = read_json(self.root / "cost-plan-fallback.json")
        self.assertNotIn("conditional_backend_estimate", actual_plan)
        self.assertEqual(actual_plan["measured_costs_seconds"]["v100"], 24.7081)
        self.assertEqual(self.experiment.workers["v100"].backend, "math")
        self.assertTrue((self.root / "units/probe-fallback-efficient-1.json").exists())

    def test_slower_fresh_candidate_raises_conditional_cost_and_can_stop_before_second_pair(self):
        summary, correctness, _ = self.conditional_fallback_probe(candidate_seconds=20.)
        self.assertEqual(summary["status"], "budget_stopped")
        correctness.assert_not_called()
        plan = read_json(self.root / "cost-plan-fallback-conditional-1.json")
        self.assertEqual(plan["conditional_backend_estimate"]["used_v100_seconds"], 20.)
        self.assertTrue((self.root / "units/probe-fallback-efficient-0.json").exists())
        self.assertFalse((self.root / "units/probe-fallback-efficient-1.json").exists())
        self.assertEqual(self.experiment.workers["v100"].backend, "math")

    def test_candidate_advisory_rejects_wrong_shape_backend_devices_or_invalid_cost_before_work(self):
        original = dict(validated=True, shape=copy.deepcopy(self.config["shapes"]["fallback"]),
                        backend="efficient", v100_seconds=13., physical_devices=copy.deepcopy(self.devices))
        for field, value in (("validated", False), ("shape", self.config["shapes"]["main"]),
                             ("backend", "math"), ("physical_devices", list(reversed(self.devices))),
                             ("v100_seconds", 0), ("v100_seconds", -1), ("v100_seconds", True),
                             ("v100_seconds", float("nan")), ("v100_seconds", float("inf"))):
            seed = self.planning_seed(dict(a100=22., v100=174.))
            seed["fallback_backend_cost_estimate"] = dict(original, **{field: value})
            self.experiment.job["planning_seed"] = seed
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "runner-validated"):
                self.experiment.probe()
            self.assertFalse(self.events)

    def test_recorded_slow_main_cost_skips_backend_units_before_falling_back(self):
        self.experiment.job["phase"] = "probe"
        self.experiment.started_monotonic = self.clock.now
        self.experiment.deadline = self.clock.now + 400.
        self.experiment.job["phase_remaining_gpu_seconds"] = dict(
            correctness=800., calibration=10000., migration=10000., mechanism=10000., reserve=2000.)
        original = self.experiment.generate

        def measured(*args, **kwargs):
            row = original(*args, **kwargs)
            if self.experiment.shape == self.config["shapes"]["main"]:
                row["elapsed_ms"] = {"a100": 22.0993, "v100": 174.2109}[kwargs.get("role", "a100")] * 1000
            return row

        with patch.object(self.experiment, "generate", side_effect=measured), \
                patch.object(self.experiment, "correctness") as correctness:
            summary = self.experiment.probe()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["shape"], self.config["shapes"]["fallback"])
        self.assertFalse((self.root / "units/probe-main-efficient-0.json").exists())
        self.assertTrue((self.root / "units/probe-fallback-efficient-0.json").exists())
        self.assertEqual(read_json(self.root / "shape-main.json")["status"], "budget_infeasible")
        correctness.assert_called_once_with(self.item)

    def test_validated_cost_seed_skips_only_infeasible_main_and_remeasures_fallback(self):
        self.experiment.job.update(phase="probe", remaining_gpu_seconds=10000.,
            planning_seed=self.planning_seed(dict(a100=22.0993, v100=174.2109)))
        with patch.object(self.experiment, "correctness") as correctness:
            summary = self.experiment.probe()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["shape"], self.config["shapes"]["fallback"])
        prepared = [args["spec"] for _, command, args, _ in self.events if command == "prepare"]
        self.assertTrue(prepared)
        self.assertTrue(all(spec["height"] == self.config["shapes"]["fallback"]["height"] for spec in prepared))
        self.assertFalse((self.root / "units/probe-main-baselines.json").exists())
        for unit in ("fallback-baselines", "fallback-efficient-0", "fallback-efficient-1"):
            self.assertEqual(read_json(self.root / "units" / ("probe-" + unit + ".json"))["status"], "completed")
        decision = read_json(self.root / "shape-main.json")
        self.assertFalse(decision["qualification_reused"])
        self.assertEqual(decision["planning_seed"]["source_unit_sha256"], "sealed-main-baselines")
        correctness.assert_called_once_with(self.item)

    def test_feasible_advisory_seed_does_not_skip_new_main_measurements(self):
        self.experiment.job.update(phase="probe", planning_seed=self.planning_seed(dict(a100=.01, v100=.02)))
        with patch.object(self.experiment, "correctness"):
            summary = self.experiment.probe()
        self.assertEqual(summary["shape"], self.config["shapes"]["main"])
        self.assertTrue((self.root / "units/probe-main-baselines.json").exists())
        self.assertFalse((self.root / "shape-main.json").exists())

    def test_unvalidated_planning_seed_cannot_control_shape_selection(self):
        seed = self.planning_seed(dict(a100=22., v100=174.))
        seed["validated"] = False
        self.experiment.job["planning_seed"] = seed
        with self.assertRaisesRegex(ValueError, "runner-validated"):
            self.experiment.probe()
        self.assertFalse(self.events)

    def test_late_pair_admission_failure_falls_back_without_inheriting_pair_estimate(self):
        self.experiment.job["phase"] = "probe"
        initial = self.experiment.job["unit_estimate_seconds"]
        original = self.experiment.unit
        admitted = []

        @contextlib.contextmanager
        def unit(name, estimate=None):
            admitted.append((name, estimate))
            if name == "main-efficient-0":
                raise BudgetStop("next complete pair cannot fit")
            with original(name, estimate):
                yield

        with patch.object(self.experiment, "unit", side_effect=unit), \
                patch.object(self.experiment, "correctness"):
            summary = self.experiment.probe()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["shape"], self.config["shapes"]["fallback"])
        self.assertEqual(dict(admitted)["fallback-baselines"], initial)
        self.assertEqual(read_json(self.root / "shape-main.json")["reason"], "complete_backend_pair_not_admitted")

    def test_unsupported_efficient_backend_keeps_math_and_finishes_probe(self):
        worker = self.experiment.workers["v100"]
        original = worker.call

        def operation(command, **args):
            if command == "advance" and worker.backend == "efficient":
                raise RemoteError(dict(error="No available kernel. Aborting execution.", error_type="RuntimeError"))
            return original(command, **args)

        worker.call = operation
        self.experiment.job["phase"] = "probe"
        with patch("ditserve.research.dual_experiment.fits_plan", return_value=True), \
                patch.object(self.experiment, "correctness"):
            summary = self.experiment.probe()
        self.assertEqual(summary["backends"]["v100"], "math")
        self.assertEqual(worker.backend, "math")
        self.assertFalse(worker.states)

    def test_failed_development_quality_keeps_all_eight_results_and_never_runs_holdout(self):
        atomic_json(self.root / "calibrate.json", dict(cache_gates={role: dict(eligible=True) for role in ("a100", "v100")}))

        def quality(candidate, reference):
            value = bound_fake_quality(candidate, reference)
            if candidate["identity"]["recipe"] == "t008":
                value.update(lpips_mean=.06, lpips_per_frame=[.06] * 17)
            return value

        self.experiment.quality = quality
        summary = self.experiment.migration()
        self.assertEqual(summary["status"], "quality_stopped")
        import json
        rows = [json.loads(line) for line in (self.root / "records.jsonl").read_text().splitlines()]
        observed = [row for row in rows if row["type"] == "migration_quality"]
        self.assertEqual(len(observed), 8)
        self.assertEqual({row["split"] for row in observed}, {"dev"})

    def test_deferred_archives_flush_after_all_decodes_and_do_not_change_makespan(self):
        for action in ("stay", "full_migrate"):
            with self.subTest(action=action):
                self.events.clear()
                self.experiment.workers = {role: DeferredFakeWorker(role, self.clock, self.events)
                                           for role in ("a100", "v100")}
                row = self.experiment.generate(self.item, "t008", role="v100", action=action, queue=2)
                latest_decode = max(result["decoded_at_monotonic"] for result in [row] + row["queued_outputs"])
                start = next(at for _, command, _, at in self.events if command == "prepare")
                boundary = next(at for _, command, _, at in self.events if command == "batch-start")
                self.assertAlmostEqual(row["elapsed_ms"], (latest_decode - start) * 1000)
                self.assertAlmostEqual(row["critical_path_ms"], (latest_decode - boundary) * 1000)
                self.assertGreaterEqual(self.clock.now - latest_decode, 1.5 - 1e-8)
                self.assertEqual(row["artifact_io_sum_ms"], 1500.)
                flushes = [at for _, command, _, at in self.events if command == "flush"]
                self.assertTrue(flushes)
                self.assertTrue(all(at >= latest_decode for at in flushes))
                commands = [command for _, command, _, _ in self.events]
                self.assertGreater(commands.index("flush"), commands.index("finish"))
                self.assertGreater(commands.index("flush"), commands.index("batch-result"))
                self.assertTrue(all(result["artifacts_pending"] is False for result in [row] + row["queued_outputs"]))
                self.assertTrue(all(not worker.pending_artifacts for worker in self.experiment.workers.values()))

    def test_attention_identity_records_actual_worker_backend_and_migration_path(self):
        self.experiment.workers["v100"].call("backend", attention="efficient")
        migrated = self.experiment.generate(self.item, role="v100", action="full_migrate")
        self.assertEqual(migrated["identity"]["attention_path"], ["efficient", "flash"])
        self.experiment.workers["a100"].call("backend", attention="math")
        control = self.experiment.generate(self.item, role="a100")
        self.assertEqual(control["identity"]["attention_path"], ["math"])


if __name__ == "__main__":
    unittest.main()
