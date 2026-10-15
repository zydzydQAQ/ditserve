"""Charged single-GPU pilot measurements, executed in a disposable process.

No simulated time is accepted here. Full videos, per-step paths, state hashes
for correctness, and raw timing records are retained before any selection.
"""

import copy
import dataclasses
import enum
import hashlib
import json
from pathlib import Path
import statistics
import time

from .budget import BudgetStop
from .common import atomic_json, digest, file_digest, read_json


def state_digest(worker, state):
    """Value-based full sampler/cache/RNG digest for instrumented correctness."""
    def visit(value):
        if worker.torch.is_tensor(value):
            raw = value.detach().cpu().contiguous().reshape(-1).view(worker.torch.uint8).numpy().tobytes()
            return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(raw).hexdigest()}
        if dataclasses.is_dataclass(value):
            return visit(vars(value))
        if isinstance(value, dict):
            return {str(k): visit(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [visit(v) for v in value]
        if isinstance(value, enum.Enum):
            return str(value)
        if value is None or type(value) in (str, bool, int, float):
            return value
        if type(value).__module__ == "torch":
            return str(value)
        raise TypeError("unsupported correctness state " + type(value).__name__)
    return digest({"payload": visit(worker._payload(state)), "step_index": state.step_index,
                   "needs_anchor": state.needs_anchor, "current_step_refreshed": state.current_step_refreshed})


def path_signature(events):
    return [{key: event.get(key) for key in ("step_index", "path", "reused_forwards", "cache_contexts")}
            for event in events if event.get("kind") == "step" or "step_index" in event]


class Experiment:
    def __init__(self, job):
        self.job = job
        self.root = Path(job["output"])
        self.frozen = read_json(self.root / "protocol.json")
        self.config = self.frozen["config"]
        self.records = []
        self.worker = self.adapter = None
        self.prompt_bank = None
        self._recorded_prompts = set()
        self.deadline = job["deadline_monotonic"]
        self.counter = 0

    def check_unit(self, estimate=None):
        if estimate is None:
            estimate = self.config["estimate_wall_seconds"][self.job["stage"]]
            plan_path = self.root / "cost-plan.json"
            if plan_path.exists():
                from .cost_planning import read_bound_cost_plan
                plan = read_bound_cost_plan(self.root, self.frozen)
                estimate = plan.get("unit_estimates", {}).get(self.job["stage"], estimate)
        if time.monotonic() + estimate >= self.deadline:
            raise BudgetStop("next complete paired unit cannot fit; unmeasured cells stay unverified")

    def record(self, record):
        record = dict(record, protocol_sha256=self.frozen["sha256"], source_sha256=self.frozen["source_sha256"],
                      device=self.job["physical_devices"][0], recorded_at=time.time())
        record["sha256"] = digest(record)
        with (self.root / "records.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
        self.records.append(record)
        print(json.dumps({key: record.get(key) for key in ("type", "case_id", "status")}), flush=True)
        return record

    def load(self):
        from .adapter import ResearchWanAdapter
        self.adapter = ResearchWanAdapter("cuda:0", model_path=self.job["model_path"], validation_only=True)
        self.worker = self.adapter.worker
        if self.job["stage"] != "correctness":
            from .prompt_bank import FrozenPromptBank
            self.prompt_bank = FrozenPromptBank(self.worker)

    def text_identity(self, spec):
        if self.prompt_bank is None:
            return {"mode": "encode_each_execution"}
        identity = self.prompt_bank.warm(spec)
        key = identity["input_sha256"]
        if key not in self._recorded_prompts:
            entry = next(row for row in self.prompt_bank.manifest() if row["input_sha256"] == key)
            self.record(dict(type="text_encoding", case_id=key, status="completed", **entry,
                             timing_scope="stage_setup_charged_to_GPU_ledger_not_per_request_mechanism"))
            self._recorded_prompts.add(key)
        return identity

    def spec(self, prompt, seed, shape, recipe):
        self.counter += 1
        return dict(shape, request_id="research-%s-%d" % (self.job["stage"], self.counter),
                    prompt=prompt["text"], negative_prompt="", seed=seed, guidance_scale=5.,
                    **{k: recipe[k] for k in ("cache", "threshold", "max_continuous")})

    def identity(self, spec, prefix, action="none", suffix=None, at_step=16):
        def recipe(value):
            return {k: value[k] for k in ("cache", "threshold", "max_continuous")}
        return dict(model_id=self.config["model_id"], revision=self.config["revision"],
                    backend={"attention": "flash", "implementation": "cache-dit-native-ulysses",
                             "gpu_name": self.job["physical_devices"][0]["name"]},
                    precision=self.config["precisions"],
                    input=dict({k: spec[k] for k in ("prompt", "negative_prompt", "seed")},
                               text_encoding=self.text_identity(spec)),
                    spec={k: spec[k] for k in ("height", "width", "frames", "steps", "guidance_scale")},
                    prefix=recipe(prefix), action={"kind": action, "at_step": 0 if action == "none" else at_step},
                    suffix=recipe(suffix or prefix), layout_history=[{"at_step": 0, "ranks": [0], "epoch": 0}],
                    protocol_sha256=self.frozen["sha256"], source_sha256=self.frozen["source_sha256"])

    def generate(self, spec, *, action="none", suffix=None, at_step=16,
                 instrument=False, urgent_spec=None, find_exact_boundary=False, capture_inventory=False,
                 admission=None):
        import numpy as np
        text_identity = self.text_identity(spec)
        if urgent_spec is not None:
            # Resolve BOTH texts before the measured interruption. Otherwise
            # the first candidate would pay the urgent bank's lazy encoding
            # inside its critical path while later candidates reuse it.
            self.text_identity(urgent_spec)
        started = time.perf_counter()
        state = self.adapter.prepare(spec) if self.prompt_bank is None else self.prompt_bank.prepare(spec)
        traces, inventories, action_records = [], [], []
        urgent = None
        applied = action == "none"
        completed_at = 0 if applied else None
        archive_overhead_ms = 0.
        critical_started = None
        try:
            while state.step_index < spec["steps"]:
                if time.monotonic() >= self.deadline:
                    raise BudgetStop("execution exceeded estimate; incomplete unit is not eligible")
                record = self.adapter.step(state)
                if instrument:
                    traces.append(state_digest(self.worker, state))
                if capture_inventory and state.step_index in (8, 16, 24, 31):
                    inventories.append({"completed_steps": state.step_index,
                                        "inventory": self.adapter.inventory(state),
                                        "memory": self.worker._memory()})
                certificate = None
                boundary = state.step_index == at_step
                if not applied and boundary:
                    critical_started = time.perf_counter()
                if action == "exact_refresh" and not applied and (boundary or find_exact_boundary):
                    certificate_started = time.perf_counter()
                    certificate = self.adapter.certificate(state)
                    action_records.append(dict(kind="refresh_certificate", ms=(time.perf_counter() - certificate_started) * 1000,
                                               completed_steps=state.step_index, eligible=certificate["eligible"],
                                               reasons=certificate["reasons"]))
                    if find_exact_boundary:
                        boundary = state.step_index >= 8 and state.step_index < spec["steps"] and certificate["eligible"]
                        if boundary:
                            critical_started = certificate_started
                if not applied and boundary:
                    if action == "exact_refresh" and not certificate["eligible"]:
                        raise IneligibleAction("no valid exact-refresh certificate at the frozen boundary: "
                                               + "; ".join(certificate["reasons"]))
                    if admission is not None:
                        admission.update(before=dict(self.worker._memory(), measurement_kind="measured_gpu"),
                                         action=action, completed_steps=state.step_index, at_step=state.step_index,
                                         artificially_filled_memory=False, single_request_oom=False)
                    action_records.append(self.adapter.snapshot(state, mode=action, certificate=certificate)
                                          if action == "exact_refresh" else self.adapter.snapshot(state, mode=action))
                    if admission is not None:
                        admission.update(after=dict(self.worker._memory(), measurement_kind="measured_gpu"),
                                         transfer_complete=True)
                    if urgent_spec is not None:
                        try:
                            urgent = self.generate(urgent_spec)
                        except self.worker.torch.OutOfMemoryError as error:
                            if admission is None:
                                raise
                            admission.update(urgent_capacity_failed=True, error=str(error),
                                             phase="urgent",
                                             failure_scope="real_urgent_request_while_main_paused")
                            raise AdmissionCapacityError(dict(admission)) from None
                        if admission is not None:
                            admission.update(urgent_allocation_succeeded=True, urgent_compute_completed=True,
                                             urgent_decode_completed=True)
                        # Raw artifact writes are evidence collection, not
                        # serving work. The urgent arm reports its measured
                        # execution before those writes and request cleanup.
                        archive_overhead_ms += urgent["artifact_io_ms"]
                    action_records.append(self.adapter.restore(state))
                    if suffix is not None and any(spec[k] != suffix[k] for k in ("cache", "threshold", "max_continuous")):
                        self.adapter.set_approximation(state, threshold=suffix["threshold"], cache_enabled=suffix["cache"])
                    applied, completed_at = True, state.step_index
            if not applied:
                reasons = sorted({reason for entry in action_records for reason in entry.get("reasons", [])})
                raise IneligibleAction("no certified refresh boundary in this trajectory: " + "; ".join(reasons))
            latent = state.latents.detach().cpu().numpy().copy()
            output = self.adapter.finish(state)
            if admission is not None:
                admission["main_completed"] = True
            elapsed_ms = (time.perf_counter() - started) * 1000 - archive_overhead_ms
            critical_path_ms = (time.perf_counter() - critical_started) * 1000 - archive_overhead_ms if critical_started is not None else None
            events = copy.deepcopy(state.events)
            artifact = self.root / "arrays" / spec["request_id"]
            archive_start = time.perf_counter()
            artifact.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(artifact) + "-video.npy", output, allow_pickle=False)
            np.save(str(artifact) + "-latent.npy", latent, allow_pickle=False)
            output_sha256 = file_digest(str(artifact) + "-video.npy")
            artifact_io_ms = (time.perf_counter() - archive_start) * 1000
            return dict(status="completed", output=output, latent=latent,
                        output_path=str(artifact.relative_to(self.root)) + "-video.npy",
                        output_sha256=output_sha256, artifact_io_ms=artifact_io_ms,
                        latent_path=str(artifact.relative_to(self.root)) + "-latent.npy",
                        elapsed_ms=elapsed_ms, traces=traces, events=events, inventories=inventories,
                        critical_path_ms=critical_path_ms,
                        timing_scope="prepare_pause_urgent_restore_suffix_decode_excluding_artifact_io",
                        text_encoding=text_identity,
                        instrumentation="step_state_hashes" if instrument else "state_inventory" if capture_inventory else "none",
                        action_records=action_records, action_step=completed_at, urgent=urgent,
                        phase_cost_ms={"prepare": state.prepare_ms,
                                       "denoise": sum(e["ms"] for e in events if e["kind"] == "step"),
                                       "decode": sum(e["ms"] for e in events if e["kind"] == "decode")})
        finally:
            self.adapter.release(state)

    @staticmethod
    def serializable(run):
        return {k: v for k, v in run.items() if k not in ("output", "latent", "urgent")}

    def quality(self, run, reference, identity, reference_identity):
        from ..gpu.quality import evaluate_quality
        from .evidence import trajectory_evidence
        metrics = evaluate_quality(reference["output"], run["output"], with_lpips=True, lpips_device="cpu")
        return trajectory_evidence(identity, metrics, correctness="passed", completed_steps=32,
                                   reference_output_sha256=reference["output_sha256"],
                                   candidate_output_sha256=run["output_sha256"],
                                   reference_identity=reference_identity)

    def correctness(self):
        import numpy as np
        from .prompt_bank import FrozenPromptBank
        frozen_bank = FrozenPromptBank(self.worker)
        shape = self.config["shapes"]["correctness"]
        prompt = self.config["prompts"][0]
        off, _, cached = self.config["recipes"]
        passed, exact_tested = True, False
        for recipe in (off, cached):
            self.check_unit()
            spec = self.spec(prompt, 101, shape, recipe)
            official = self.worker.reference(spec)
            baseline = self.generate(self.spec(prompt, 101, shape, recipe), instrument=True)
            official_equal = (np.array_equal(official["output"], baseline["output"])
                              and np.array_equal(official["latents"].cpu().numpy(), baseline["latent"]))
            self.record(dict(type="correctness", case_id="official-" + recipe["id"],
                             status="passed" if official_equal else "failed", official_equal=bool(official_equal)))
            if not official_equal:
                raise RuntimeError("stepwise worker differs from official pipeline")
            self.prompt_bank = frozen_bank
            try:
                frozen_candidate = self.generate(self.spec(prompt, 101, shape, recipe), instrument=True)
            finally:
                self.prompt_bank = None
            frozen_equal = (np.array_equal(frozen_candidate["output"], baseline["output"])
                and np.array_equal(frozen_candidate["latent"], baseline["latent"])
                and frozen_candidate["traces"] == baseline["traces"]
                and path_signature(frozen_candidate["events"]) == path_signature(baseline["events"]))
            self.record(dict(type="correctness", case_id="frozen-text-" + recipe["id"],
                             status="passed" if frozen_equal else "failed", exact_match=bool(frozen_equal)))
            if not frozen_equal:
                raise RuntimeError("frozen text installation changed the complete official-equivalent trajectory")
            for action in ("keep", "hot", "exact_refresh", "cold"):
                if action == "exact_refresh" and not recipe["cache"]:
                    continue
                try:
                    candidate = self.generate(self.spec(prompt, 101, shape, recipe), action=action,
                                              instrument=True, find_exact_boundary=action == "exact_refresh",
                                              urgent_spec=self.spec(prompt, 102, shape, off))
                except IneligibleAction as error:
                    self.record(dict(type="correctness", case_id=recipe["id"] + "-" + action,
                                     status="unverified", reason=str(error)))
                    continue
                exact = (np.array_equal(candidate["output"], baseline["output"])
                         and np.array_equal(candidate["latent"], baseline["latent"])
                         and candidate["traces"] == baseline["traces"]
                         and path_signature(candidate["events"]) == path_signature(baseline["events"]))
                if action != "cold":
                    passed = passed and exact
                    exact_tested = exact_tested or (action == "exact_refresh" and exact)
                self.record(dict(type="correctness", case_id=recipe["id"] + "-" + action,
                                 status="passed" if exact or action == "cold" else "failed",
                                 check="structural_only_approximate" if action == "cold" else "full_step_state_equality",
                                 exact_match=bool(exact), action=action, recipe=recipe,
                                 candidate=self.serializable(candidate), baseline=self.serializable(baseline)))
                if not passed:
                    raise RuntimeError("exact same-layout restore changed trajectory")
        atomic_json(self.root / "correctness.json", dict(status="passed" if passed else "failed",
                    exact_refresh="passed" if exact_tested else "unverified", protocol_sha256=self.frozen["sha256"]))

    def probe(self):
        from ..gpu.quality import evaluate_quality
        from .cost_planning import choose_cost_plan
        off, _, cached = self.config["recipes"]
        completed, costs = [], []
        stop_reason = None
        for frames in self.config["frame_candidates"]:
            try:
                self.check_unit()
            except BudgetStop as error:
                stop_reason = str(error)
                break
            shape = dict(self.config["shapes"]["probe"], frames=frames)
            pair, videos = [], []
            try:
                for recipe in (off, cached):
                    resident_bytes = self.worker.torch.cuda.memory_allocated(self.worker.device)
                    self.worker.torch.cuda.reset_peak_memory_stats(self.worker.device)
                    run = self.generate(self.spec(self.config["prompts"][0], 101, shape, recipe), capture_inventory=True)
                    videos.append(run["output"])
                    pair.append(dict(recipe=recipe, run=self.serializable(run),
                                     resident_allocated_bytes=resident_bytes,
                                     peak_allocated_bytes=self.worker.torch.cuda.max_memory_allocated(self.worker.device),
                                     peak_reserved_bytes=self.worker.torch.cuda.max_memory_reserved(self.worker.device)))
            except self.worker.torch.OutOfMemoryError as error:
                self.record(dict(type="probe", case_id=str(frames), status="capacity_failed", frames=frames,
                                 reason=str(error), complete_pair=False,
                                 interpretation="single_request_capacity_failure_is_not_state_management_benefit"))
                break
            except BudgetStop as error:
                stop_reason = str(error)
                self.record(dict(type="probe", case_id=str(frames), status="incomplete_budget_stop",
                                 frames=frames, complete_pair=False, reason=stop_reason))
                break
            quality_started = time.perf_counter()
            quality_measurement = evaluate_quality(videos[0], videos[1], with_lpips=True, lpips_device="cpu")
            quality_seconds = time.perf_counter() - quality_started
            costs.append(dict(shape=shape, measurement_kind="measured_gpu",
                generation_wall_seconds=max((arm["run"]["elapsed_ms"] + arm["run"]["artifact_io_ms"]) / 1000 for arm in pair),
                quality_wall_seconds=quality_seconds if quality_measurement.get("status") == "ok" else None,
                quality_measurement_status=quality_measurement.get("status"),
                transfer_reservation_seconds=1.,
                timing_scope=dict(encoding_included=False, encoding_mode="frozen_text_stage_setup_reserved_separately", archive_included=True),
                source_sha256=self.frozen["source_sha256"]))
            self.record(dict(type="probe", case_id=str(frames), status="completed", shape=shape, arms=pair,
                             measurement_kind="measured_gpu", scope="mechanism_not_online_e2e"))
            completed.append(shape)
            del videos, run
        ledger = read_json(self.frozen["ledger"])
        remaining = ledger["stage_limits_gpu_seconds"]["calibrate"] - ledger["stage_used_gpu_seconds"]["calibrate"]
        from ..benchmark.common import seal
        plan = seal(dict(choose_cost_plan(costs, remaining_calibrate_seconds=remaining),
                         protocol_sha256=self.frozen["sha256"], source_sha256=self.frozen["source_sha256"]))
        atomic_json(self.root / "cost-plan.json", plan)
        atomic_json(self.root / "probe.json", dict(status="completed" if plan.get("selected_shape") else plan["status"],
                    selected_shape=plan.get("selected_shape"), completed_shapes=completed, cost_plan=plan,
                    stop_reason=stop_reason,
                    protocol_sha256=self.frozen["sha256"]))

    def calibrate(self):
        from .matrix import select_initial_recipe
        shape = read_json(self.root / "probe.json")["selected_shape"]
        rows = []
        for prompt in [p for p in self.config["prompts"] if p["split"] == "dev"]:
            for seed in self.config["dev_seeds"]:
                self.check_unit()
                off = self.config["recipes"][0]
                ref_spec = self.spec(prompt, seed, shape, off)
                reference = self.generate(ref_spec)
                ref_identity = self.identity(ref_spec, off)
                for recipe in self.config["recipes"]:
                    spec = self.spec(prompt, seed, shape, recipe)
                    run = reference if not recipe["cache"] else self.generate(spec)
                    identity = self.identity(spec, recipe)
                    evidence = self.quality(run, reference, identity, ref_identity)
                    row = self.record(dict(type="calibration", case_id=spec["request_id"],
                        recipe_id=recipe["id"], prompt_id=prompt["id"], seed=seed, split="dev", status="completed",
                        measurement_kind="measured_gpu", phase_cost_ms=run["phase_cost_ms"],
                        evidence=evidence, run=self.serializable(run)))
                    rows.append(row)
        selection = select_initial_recipe(rows, self.config, selected_shape=shape,
                                          protocol_sha256=self.frozen["sha256"])
        atomic_json(self.root / "calibration.json", selection)

    def mechanism(self):
        # The bounded screening/holdout driver is below; four-GPU execution has
        # its own gate and never falls back to a heterogeneous pair.
        return run_state_mechanism(self)


class IneligibleAction(RuntimeError):
    pass


class AdmissionCapacityError(RuntimeError):
    def __init__(self, observation):
        self.observation = observation
        super().__init__("real urgent request exhausted capacity while main request was paused")


def run_state_mechanism(experiment):
    """Screen on development costs, validate <=2 policies, then confirm holdout.

    Quality cells run once. Counterbalanced timing pairs are a separate bounded
    experiment over the first and last already-probed shapes. No synthetic
    pressure or inferred memory-admission result is constructed here.
    """
    from ..benchmark.common import verify_seal
    from .evidence import qualify_trajectory, research_gates
    from .matrix import build_matrix, materialize_joint_jobs, select_joint_candidates
    exp, config = experiment, experiment.config
    initial = verify_seal(read_json(exp.root / "calibration.json"))
    selected_initial = initial.get("selected")
    if initial.get("status") != "selected" or not isinstance(selected_initial, dict):
        raise IneligibleAction("no complete development initial-strategy qualification")
    if initial.get("protocol_sha256") != exp.frozen["sha256"]:
        raise IneligibleAction("initial strategy belongs to another frozen protocol")
    prefix = next(r for r in config["recipes"] if r["id"] == selected_initial["recipe_id"])
    probe = read_json(exp.root / "probe.json")
    shape = probe["selected_shape"]
    correctness = read_json(exp.root / "correctness.json")
    if correctness.get("status") != "passed" or correctness.get("protocol_sha256") != exp.frozen["sha256"]:
        raise IneligibleAction("same-protocol correctness is required before action screening")
    exact = correctness.get("exact_refresh") == "passed"
    actions = [action for action in config["actions"] if action != "exact_refresh" or exact]
    matrix = build_matrix(config, shape, protocol_sha256=exp.frozen["sha256"],
                          initial_recipe_id=prefix["id"], allowed_actions=actions)
    atomic_json(exp.root / "mechanism-matrix.json", matrix)
    off = next(r for r in config["recipes"] if r["id"] == "off")
    prompts = {p["id"]: p for p in config["prompts"]}
    dev = [p for p in config["prompts"] if p["split"] == "dev"]
    urgent_prompt, urgent_seed = dev[1], config["dev_seeds"][1]
    receipt = dict(status="running", protocol_sha256=exp.frozen["sha256"], matrix_sha256=matrix["sha256"],
                   screen=[], development=[], holdout=[], state_trials=[], memory_trials=[],
                   selected_candidates=[], development_qualified_candidates=[], holdout_confirmed_candidates=[],
                   holdout_used_for_selection=False, gates=research_gates(),
                   missing_measurements=["Natural keep capacity rejection followed by completed real admission and qualified resume; memory admission is unmeasured, not disproven."])

    def persist():
        receipt["gates"] = research_gates(state_trials=receipt["state_trials"], memory_trials=receipt["memory_trials"])
        atomic_json(exp.root / "mechanism.json", receipt)

    def critical_cost(run):
        value = run.get("critical_path_ms")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise RuntimeError("state comparison requires measured step-boundary-to-completion critical_path_ms")
        return value

    def measured_action(candidate, prompt, seed, cell_shape):
        spec = exp.spec(prompt, seed, cell_shape, prefix)
        urgent_spec = exp.spec(urgent_prompt, urgent_seed, cell_shape, off)
        run = exp.generate(spec, action=candidate["action"]["kind"], suffix=candidate["suffix"],
                           at_step=candidate["action"]["at_step"], urgent_spec=urgent_spec)
        identity = exp.identity(spec, prefix, candidate["action"]["kind"], candidate["suffix"],
                                at_step=candidate["action"]["at_step"])
        return run, identity, exp.identity(urgent_spec, off)

    def references(prompt, seed, cell_shape):
        spec = exp.spec(prompt, seed, cell_shape, off)
        urgent_spec = exp.spec(urgent_prompt, urgent_seed, cell_shape, off)
        return (exp.generate(spec), exp.identity(spec, off),
                exp.generate(urgent_spec), exp.identity(urgent_spec, off))

    def evaluate(run, identity, urgent_identity, reference):
        video, ref_identity, urgent_video, urgent_ref_identity = reference
        proof = exp.quality(run, video, identity, ref_identity)
        urgent_proof = exp.quality(run["urgent"], urgent_video, urgent_identity, urgent_ref_identity)
        qualified = (qualify_trajectory(proof, identity, epsilon=config["epsilon"])["qualified"] and
                     qualify_trajectory(urgent_proof, urgent_identity, epsilon=config["epsilon"])["qualified"])
        arm = dict(cost_ms=critical_cost(run), cost_scope="full_critical_path", identity=identity, evidence=proof,
                   urgent=dict(identity=urgent_identity, evidence=urgent_proof))
        return qualified, arm

    persist()
    # Screening is not quality qualification. Every predeclared candidate must
    # produce a measured cost or an explicit structural ineligibility reason.
    for job in matrix["screen_jobs"]:
        exp.check_unit()
        candidate = next(c for c in matrix["screen_candidates"] if c["candidate_id"] == job["candidate_id"])
        spec = exp.spec(prompts[job["prompt_id"]], job["seed"], shape, prefix)
        identity = exp.identity(spec, prefix, candidate["action"]["kind"], candidate["suffix"])
        context = {k: v for k, v in identity.items() if k not in ("input", "prefix", "suffix", "action")}
        row = dict(type="action_screen", case_id=job["job_id"], candidate_id=job["candidate_id"],
                   prompt_id=job["prompt_id"], seed=job["seed"], split="dev", correctness_status="passed",
                   protocol_sha256=exp.frozen["sha256"], context_sha256=digest(context), measurement_kind="measured_gpu")
        try:
            run, _, _ = measured_action(candidate, prompts[job["prompt_id"]], job["seed"], shape)
            row.update(status="completed", cost_ms=critical_cost(run), run=exp.serializable(run))
        except IneligibleAction as error:
            row.update(status="ineligible", reason=str(error))
        receipt["screen"].append(exp.record(row))
        persist()
    shortlist_limit = probe.get("cost_plan", {}).get("candidate_limit", config["max_joint_candidates"])
    selection = select_joint_candidates(receipt["screen"], matrix=matrix, limit=shortlist_limit)
    atomic_json(exp.root / "action-selection.json", selection)
    if selection["status"] != "selected_for_validation":
        receipt.update(status="insufficient_evidence", reason="development cost screen is incomplete")
        persist()
        return receipt
    chosen = {row["candidate_id"]: row["candidate"] for row in selection["selected"]}
    receipt["selected_candidates"] = list(chosen)
    persist()

    # Full development quality once per input, not two timed orders per cell.
    # The same independently generated cache-off reference can assess both
    # candidates for that exact input; model inputs are prepared per execution.
    for split in ("dev", "holdout"):
        jobs = materialize_joint_jobs(matrix, selection, split=split)
        if split == "holdout":
            jobs = [job for job in jobs if job["candidate_id"] in receipt["development_qualified_candidates"]]
        groups = {}
        for job in jobs:
            groups.setdefault((job["prompt_id"], job["seed"]), []).append(job)
        destination = "development" if split == "dev" else "holdout"
        for (prompt_id, seed), cells in groups.items():
            exp.check_unit()
            reference = references(prompts[prompt_id], seed, shape)
            for job in cells:
                exp.check_unit()
                row = dict(type="joint_quality", case_id=job["job_id"], candidate_id=job["candidate_id"],
                           split=split, prompt_id=prompt_id, seed=seed, measurement_kind="measured_gpu")
                try:
                    run, identity, urgent_identity = measured_action(chosen[job["candidate_id"]], prompts[prompt_id], seed, shape)
                    passed, arm = evaluate(run, identity, urgent_identity, reference)
                    row.update(status="completed" if passed else "unqualified", **arm, run=exp.serializable(run))
                except IneligibleAction as error:
                    row.update(status="ineligible", reason=str(error))
                receipt[destination].append(exp.record(row))
                persist()
        expected = {(row["prompt_id"], row["seed"]) for row in
                    matrix["development_inputs" if split == "dev" else "holdout_inputs"]}
        qualified_ids = []
        for candidate_id in chosen:
            cells = [row for row in receipt[destination] if row["candidate_id"] == candidate_id]
            if (len(cells) == len(expected) and {(row["prompt_id"], row["seed"]) for row in cells} == expected and
                    all(row["status"] == "completed" for row in cells)):
                qualified_ids.append(candidate_id)
        receipt["development_qualified_candidates" if split == "dev" else "holdout_confirmed_candidates"] = qualified_ids
        persist()

    # Freeze actual cost conditions from already completed probes, never infer
    # pressure from nominal bytes. One available shape remains insufficient for
    # a condition-dependent crossover claim, even if its pair is faster.
    shapes = {digest(s): s for s in probe.get("cost_plan", {}).get("paired_conditions", probe.get("completed_shapes", [shape]))}
    ordered_shapes = sorted(shapes.values(), key=lambda s: (s["frames"], s["height"], s["width"]))
    conditions = ordered_shapes[:1] + (ordered_shapes[-1:] if len(ordered_shapes) > 1 else [])
    receipt["paired_conditions"] = conditions
    baseline = dict(prefix=matrix["screen_candidates"][0]["prefix"], suffix=matrix["screen_candidates"][0]["prefix"],
                    action=dict(kind="keep", at_step=config["transition_step"]))
    for cell_shape in conditions:
        if not receipt["holdout_confirmed_candidates"]:
            break
        exp.check_unit()
        reference = references(dev[0], config["dev_seeds"][0], cell_shape)
        for candidate_id in receipt["holdout_confirmed_candidates"]:
            for repeat, order in enumerate((("keep", "candidate"), ("candidate", "keep"))):
                exp.check_unit()
                arms = {}
                for name in order:
                    candidate = baseline if name == "keep" else chosen[candidate_id]
                    try:
                        run, identity, urgent_identity = measured_action(candidate, dev[0], config["dev_seeds"][0], cell_shape)
                        _, arms[name] = evaluate(run, identity, urgent_identity, reference)
                    except IneligibleAction:
                        break
                trial = exp.record(dict(type="state_trial", case_id="pair-%s-%s-%d" % (candidate_id, cell_shape["frames"], repeat),
                    condition_id="%s-%dx%d-%df" % (candidate_id, cell_shape["height"], cell_shape["width"], cell_shape["frames"]),
                    pair_id=str(repeat), order=list(order), arms=arms, status="completed" if len(arms) == 2 else "incomplete",
                    measurement_kind="measured_gpu"))
                receipt["state_trials"].append(trial)
                persist()
    # A real capacity trial is separate from timing and never uses allocator
    # fillers. The already-selected actions remain fixed on the holdout input.
    for candidate_id in receipt["holdout_confirmed_candidates"]:
        candidate = chosen[candidate_id]
        if candidate["action"]["kind"] == "keep":
            continue
        from .memory_experiment import run_memory_admission
        memory_trial = run_memory_admission(exp, prefix, shape, candidate)
        receipt["memory_trials"].append(memory_trial)
        receipt["missing_measurements"] = [] if memory_trial.get("status") in ("completed", "no_admission_difference") else receipt["missing_measurements"]
        persist()
        if memory_trial.get("status") == "no_admission_difference" or memory_trial.get("supported"):
            break
    receipt.update(status="completed", quality_qualified=bool(receipt["holdout_confirmed_candidates"]),
                   scope="development-selected policies; holdout confirms without reselection; bounded paired state mechanism")
    persist()
    return receipt
