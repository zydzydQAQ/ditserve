"""Executable finite heterogenous SP1 experiment; no online controller."""

import contextlib
import math
from pathlib import Path
import statistics
import time

from .common import atomic_json, digest, read_json
from .budget import BudgetStop
from .dual_evidence import UnitJournal, read_records
from .dual_matrix import (ACTIONS, cache_gate, choose_queue, estimate_matrix, fits_plan, inputs,
                          known_quality, migration_quality_gate, seal, select_backend, select_migration)
from .dual_runtime import ResidentWorker, RemoteError, transfer


class IneligibleAction(RuntimeError):
    pass


class DualExperiment:
    def __init__(self, job):
        self.job, self.root = job, Path(job["output"])
        self.frozen = read_json(self.root / "protocol.json")
        self.config = self.frozen["config"]
        self.deadline, self.workers, self.counter = job["deadline_monotonic"], {}, 0
        self.started_monotonic = time.monotonic()
        self.journal = UnitJournal(self.root, self.frozen, [x["uuid"] for x in job["physical_devices"]])
        self.shape = self.config["shapes"]["main"]
        self.backends = dict(a100="flash", v100="math")
        if (self.root / "probe.json").exists():
            probe = read_json(self.root / "probe.json")
            if probe.get("status") == "completed":
                self.shape, self.backends = probe["shape"], probe["backends"]

    def write(self, name, value):
        value = {k: v for k, v in value.items() if k != "sha256"}
        value = seal(dict(value, protocol_sha256=self.frozen["sha256"], source_sha256=self.frozen["source_sha256"]))
        atomic_json(self.root / (name + ".json"), value)
        return value

    def load(self):
        for device in self.job["physical_devices"]:
            role = "a100" if "A100" in device["name"] else "v100"
            self.workers[role] = ResidentWorker(device["uuid"], self.backends[role], self.config["model_path"],
                                                 self.root, self.deadline)

    def close(self):
        failures = []
        for worker in self.workers.values():
            try:
                worker.close()
            except BaseException as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError("; ".join(failures))

    def context(self):
        return digest(dict(protocol=self.frozen["sha256"], source=self.frozen["source_sha256"],
                           shape=self.shape, backends=self.backends,
                           devices=self.job.get("all_physical_devices", self.job["physical_devices"]), transition_step=16))

    def spec(self, item, recipe="off"):
        self.counter += 1
        return dict(self.shape, request_id="dual-%s-%06d" % (self.job["phase"], self.counter),
                    prompt=item["prompt"], negative_prompt="", seed=item["seed"], guidance_scale=5.,
                    cache=recipe == "t008", threshold=.08 if recipe == "t008" else 0., max_continuous=1)

    def unit(self, name, estimate=None):
        return self.journal.unit(self.job["phase"] + "-" + name,
                                 estimate or self.job["unit_estimate_seconds"], self.deadline)

    def record(self, type_, case_id, **row):
        context = row.pop("context_sha256", self.context())
        if context != self.context():
            raise ValueError("trajectory context differs from the active frozen configuration")
        return self.journal.append(dict(type=type_, case_id=case_id, context_sha256=context,
                                        measurement_kind="measured_gpu", **row))

    def generate(self, item, recipe="off", *, role="a100", action="stay", instrument=False, queue=0):
        spec = self.spec(item, recipe)
        source = self.workers[role]
        target = self.workers["a100"] if action in ACTIONS[1:] or queue else source
        source.call("warm", spec=spec)
        queue_specs = [self.spec(inputs(self.config)[0], "off") for _ in range(queue)]
        for queued in queue_specs:
            target.call("warm", spec=queued)
        start = time.monotonic()
        prepared = source.call("prepare", spec=spec)
        request_id = spec["request_id"]
        owner, live, queued_outputs = source, True, []
        try:
            prefix = source.call("advance", request_id=request_id, until=16, instrument=instrument)
            checkpoint = source.call("checkpoint", request_id=request_id) if instrument else {}
            cert = source.call("certificate", request_id=request_id) if action.endswith("certified_omit") else None
            if cert is not None and not cert["eligible"]:
                raise IneligibleAction("step16 source certificate: " + "; ".join(cert["reasons"]))
            boundary = time.monotonic()
            if queue:
                target.start("batch", specs=queue_specs)
            migration = None
            if action.startswith("local_"):
                migration = source.call("local_roundtrip", request_id=request_id,
                                        action=action[len("local_"):])
                wait_ms = 0.
            elif action != "stay":
                if role != "v100":
                    raise ValueError("only V100 to A100 migration is permitted")
                if queue:
                    queued_outputs = target.result()
                wait_ms = (time.monotonic() - boundary) * 1000
                migration = transfer(source, target, request_id, action, request_id + "-migration", cert)
                owner = target
            else:
                wait_ms = 0.
            suffix = owner.call("advance", request_id=request_id, until=32, instrument=instrument)
            run = owner.call("finish", request_id=request_id, retain_latent=instrument, defer_artifacts=True)
            live = False
            if queue and action == "stay":
                queued_outputs = target.result()
            decoded = max(r.get("decoded_at_monotonic", time.monotonic()) for r in [run] + queued_outputs)
            elapsed, critical = (decoded - start) * 1000, (decoded - boundary) * 1000
            # Persist only after all participating requests have decoded. This
            # avoids adding archive writes to queue waiting or subtracting
            # overlapping I/O from the measured wall-clock critical path.
            for resident in dict.fromkeys([owner] + ([target] if queue else [])):
                pending = [r for r in [run] + queued_outputs if r.get("artifacts_pending")
                           and (resident is owner if r is run else resident is target)]
                if pending:
                    saved = resident.call("flush", request_ids=[r["request_id"] for r in pending])
                    for r in pending:
                        r.update(saved[r["request_id"]])
            artifact_ms = run["artifact_io_ms"] + sum(q["artifact_io_ms"] for q in queued_outputs)
            run.update(elapsed_ms=elapsed, critical_path_ms=critical, target_wait_ms=wait_ms,
                       action=action, action_step=16, recipe_id=recipe, device_role=role,
                       prompt_id=item["prompt_id"], seed=item["seed"], split=item["split"],
                       migration=migration, queued_outputs=queued_outputs,
                       all_requests_decoded=True, text_encoding=prepared["text_encoding"],
                       traces=prefix["traces"] + suffix["traces"],
                       inventories=prefix["inventories"] + suffix["inventories"],
                       checkpoint=checkpoint, artifact_io_sum_ms=artifact_ms,
                       timing_scope="frozen_text_mechanism_wall_including_RPC_transfer_wait_all_decode; artifact_IO_after_measurement",
                       identity=dict(context_sha256=self.context(), input=item, recipe=recipe, action=action,
                                     path=[role] if action == "stay" or action.startswith("local_") else ["v100", "a100"],
                                     attention_path=[getattr(source, "backend", self.backends[role])]
                                         + ([getattr(owner, "backend", self.backends["a100"])] if owner is not source else []),
                                     boundary=16))
            return run
        finally:
            if live and not owner.pending:
                with contextlib.suppress(Exception):
                    owner.call("release", request_id=request_id)

    def quality(self, candidate, reference):
        import numpy as np
        from ..gpu.quality import evaluate_quality
        from .retention import verify_reference
        paths = []
        for row in (candidate, reference):
            checked = verify_reference(self.root, row["output_path"], row["output_sha256"])
            if not checked["re_evaluable"]:
                raise ValueError("new quality evaluation requires retained original arrays")
            paths.append(checked["path"])
        metrics = evaluate_quality(np.load(paths[1], allow_pickle=False),
                                   np.load(paths[0], allow_pickle=False),
                                   with_lpips=True, lpips_device="cpu")
        metrics["lpips_max"] = max(metrics["lpips_per_frame"]) if metrics["lpips_per_frame"] else None
        metrics.update(candidate_output_sha256=candidate["output_sha256"],
                       reference_output_sha256=reference["output_sha256"],
                       candidate_identity_sha256=digest(candidate["identity"]),
                       reference_identity_sha256=digest(reference["identity"]))
        return metrics

    @staticmethod
    def _backend_pair_wall(costs, previous_pairs=()):
        # A pair contains two trajectories. Until measured, charge EFFICIENT as
        # MATH; retain any slower observed candidate cost in later admission.
        candidate = max([costs["v100"]] + [p["costs_ms"]["efficient"] / 1000 for p in previous_pairs])
        return (costs["v100"] + candidate) * 1.3 + 30.

    def _probe_cost_plan(self, costs, *, backend_pairs=0, pair_wall=None, baselines_pending=False,
                         candidate_estimate=None, previous_pairs=()):
        future_costs = dict(costs)
        if candidate_estimate is not None:
            future_costs["v100"] = max([candidate_estimate["v100_seconds"]]
                + [pair["costs_ms"]["efficient"] / 1000 for pair in previous_pairs])
        plan = estimate_matrix(future_costs)
        plan.pop("sha256")
        resident = len(self.workers)
        trajectory_cost = 12 * (future_costs["a100"] + future_costs["v100"]) * 1.3 + 120
        # Qualification still admits two MATH-cost trajectories per pair, even
        # when future work is conditionally estimated using a prior candidate.
        backend_cost = backend_pairs * (self._backend_pair_wall(costs) if pair_wall is None else pair_wall) * resident
        baseline_cost = ((2 * costs["a100"] + costs["v100"]) * 1.3 + 30.) * resident if baselines_pending else 0.
        plan["correctness_components_gpu_seconds"] = dict(trajectories=trajectory_cost,
            pending_backend_pairs=backend_cost, pending_baselines=baseline_cost)
        plan["phases_gpu_seconds"]["correctness"] = trajectory_cost + backend_cost + baseline_cost
        plan["required_phases"].append("correctness")
        plan["required_gpu_seconds"] = sum(plan["phases_gpu_seconds"][stage] for stage in plan["required_phases"])
        plan["planned_with_optional_gpu_seconds"] = sum(plan["phases_gpu_seconds"].values())
        if candidate_estimate is not None:
            plan["conditional_backend_estimate"] = dict(candidate_estimate,
                used_v100_seconds=future_costs["v100"], qualification_reused=False,
                requires_fresh_backend_qualification=True)
            plan["assumptions"] += "; future candidate costs are advisory until fresh backend qualification"
        return seal(plan)

    def _fallback_backend_estimate(self, seed):
        value = seed.get("fallback_backend_cost_estimate") if seed is not None else None
        if value is None:
            return None
        if (not isinstance(value, dict) or value.get("validated") is not True
                or value.get("shape") != self.config["shapes"]["fallback"]
                or value.get("backend") != "efficient"
                or type(value.get("v100_seconds")) not in (int, float)
                or not math.isfinite(value["v100_seconds"]) or value["v100_seconds"] <= 0
                or ("physical_devices" in value and value["physical_devices"] !=
                    self.job.get("all_physical_devices", self.job["physical_devices"]))):
            raise ValueError("fallback backend estimate must be runner-validated for the identical shape and devices")
        return value

    def _probe_admission(self, plan):
        consumed = max(0., time.monotonic() - self.started_monotonic) * len(self.workers)
        remaining = max(0., self.job["remaining_gpu_seconds"] - consumed)
        phases = self.job.get("phase_remaining_gpu_seconds")
        if phases is not None:
            phases = dict(phases)
            # The runner may explicitly fund this retry from reserve. It supplies
            # the authorized allowance under this logical correctness key.
            phases["correctness"] = max(0., phases.get("correctness", 0.) - consumed)
        time_left = max(0., self.deadline - time.monotonic())
        checks_wall = plan["phases_gpu_seconds"]["correctness"] / len(self.workers)
        fits = fits_plan(plan, remaining, phases)
        return dict(eligible=fits and checks_wall < time_left,
                    reason="complete_matrix_or_phase_budget" if not fits else
                           "remaining_probe_checks_exceed_deadline" if checks_wall >= time_left else "admitted",
                    remaining_gpu_seconds=remaining, phase_remaining_gpu_seconds=phases,
                    remaining_wall_seconds=time_left, remaining_checks_wall_seconds=checks_wall)

    def _shape_budget_skip(self, shape_id, plan, admission, *, planning_seed=None):
        self.write("cost-plan-" + shape_id, plan)
        self.write("shape-" + shape_id, dict(status="budget_infeasible", reason=admission["reason"],
            admission=admission, cost_plan_sha256=plan["sha256"], planning_seed=planning_seed,
            qualification_reused=False, quality_samples_reduced=False))

    def probe(self):
        item = inputs(self.config)[0]
        initial_unit_estimate = self.job["unit_estimate_seconds"]
        seed = self.job.get("planning_seed")
        fallback_estimate = self._fallback_backend_estimate(seed)
        skip_main = False
        if seed is not None:
            if seed.get("validated") is not True or seed.get("shape") != self.config["shapes"]["main"]:
                raise ValueError("planning seed must be runner-validated main-shape advisory costs")
            seeded_plan = self._probe_cost_plan(seed["measured_costs_seconds"], backend_pairs=2, baselines_pending=True)
            admission = self._probe_admission(seeded_plan)
            if not admission["eligible"]:
                self._shape_budget_skip("main", seeded_plan, admission, planning_seed=seed)
                skip_main = True
        for shape_id in ("main", "fallback"):
            if shape_id == "main" and skip_main:
                continue
            self.shape = self.config["shapes"][shape_id]
            costs, pairs = {}, []
            try:
                with self.unit("%s-baselines" % shape_id, estimate=initial_unit_estimate):
                    baselines = {}
                    for role in ("a100", "v100"):
                        run = self.generate(item, role=role, instrument=True)
                        baselines[role] = run
                        costs[role] = run["elapsed_ms"] / 1000
                        self.record("probe", role + "-" + shape_id, status="completed", run=run)
                    # Both-MATH control is explicit and never qualifies Flash costs.
                    self.workers["a100"].call("backend", attention="math")
                    control = self.generate(item, role="a100")
                    self.record("backend_control", shape_id, status="completed", a100_math=control,
                                v100_math=baselines["v100"], quality=self.quality(control, baselines["v100"]))
            except BudgetStop as error:
                self.write("shape-" + shape_id, dict(status="budget_infeasible", reason="complete_baseline_unit_not_admitted",
                    detail=str(error), qualification_reused=False, quality_samples_reduced=False))
                continue
            except RemoteError as error:
                if error.remote_type != "OutOfMemoryError":
                    raise
                self.write("shape-" + shape_id, dict(status="capacity_failure", reason=str(error)))
                if shape_id == "fallback":
                    return self.write("probe", dict(status="capacity_stopped", reason="both frozen shapes failed real single-request capacity"))
                continue
            finally:
                self.workers["a100"].call("backend", attention="flash")
            candidate_estimate = fallback_estimate if shape_id == "fallback" else None
            preliminary = self._probe_cost_plan(costs, backend_pairs=2, candidate_estimate=candidate_estimate)
            if candidate_estimate is not None:
                self.write("cost-plan-fallback-conditional", preliminary)
            admission = self._probe_admission(preliminary)
            if not admission["eligible"]:
                self._shape_budget_skip(shape_id, preliminary, admission)
                continue
            backend_budget_failed = False
            for order_id, order in enumerate((("math", "efficient"), ("efficient", "math"))):
                pair_wall = self._backend_pair_wall(costs, pairs)
                pending_plan = self._probe_cost_plan(costs, backend_pairs=2 - order_id, pair_wall=pair_wall,
                    candidate_estimate=candidate_estimate, previous_pairs=pairs)
                if candidate_estimate is not None:
                    self.write("cost-plan-fallback-conditional-" + str(order_id), pending_plan)
                admission = self._probe_admission(pending_plan)
                if not admission["eligible"]:
                    self._shape_budget_skip(shape_id, pending_plan, admission)
                    backend_budget_failed = True
                    break
                try:
                    with self.unit("%s-efficient-%d" % (shape_id, order_id), estimate=pair_wall):
                        runs = {}
                        for backend in order:
                            self.workers["v100"].call("backend", attention=backend)
                            runs[backend] = self.generate(item, role="v100")
                        pair = dict(order=list(order), shape=self.shape, context_sha256=self.context(),
                                    performance_valid=True, quality=self.quality(runs["efficient"], runs["math"]),
                                    costs_ms={k: v["elapsed_ms"] for k, v in runs.items()})
                        self.record("backend_pair", shape_id + "-" + str(order_id), status="completed", runs=runs, **pair)
                        pairs.append(pair)
                except BudgetStop as error:
                    self._shape_budget_skip(shape_id, pending_plan,
                        dict(admission, eligible=False, reason="complete_backend_pair_not_admitted", detail=str(error)))
                    backend_budget_failed = True
                    break
                except RemoteError as error:
                    # Native kernel unavailability is an observation, not a fallback to a third backend.
                    if error.remote_type not in ("RuntimeError", "NotImplementedError") or not any(
                            word in str(error).lower() for word in ("no available kernel", "no viable backend", "not supported")):
                        raise
                    self.write("efficient-" + shape_id, dict(status="unsupported", reason=str(error)))
                    break
                finally:
                    self.workers["v100"].call("backend", attention="math")
            if backend_budget_failed:
                continue
            choice = select_backend(pairs, shape=self.shape, context_sha256=self.context())
            self.backends["v100"] = choice["selected"]
            self.workers["v100"].call("backend", attention=choice["selected"])
            if choice["selected"] == "efficient":
                costs["v100"] = max(p["costs_ms"]["efficient"] for p in pairs) / 1000
            plan = self._probe_cost_plan(costs)
            self.write("cost-plan-" + shape_id, plan)
            admission = self._probe_admission(plan)
            if admission["eligible"]:
                self.job["unit_estimate_seconds"] = max(costs.values()) * 3 * 1.3 + 30
                self.correctness(item)
                return self.write("probe", dict(status="completed", shape=self.shape, backends=self.backends,
                                                 backend_selection=choice, cost_plan=plan, context_sha256=self.context()))
            self._shape_budget_skip(shape_id, plan, admission)
            self.backends["v100"] = "math"
            self.workers["v100"].call("backend", attention="math")
        return self.write("probe", dict(status="budget_stopped", reason="complete matrix does not fit either shape; samples unchanged"))

    def correctness(self, item):
        # Target-local restore must preserve each step. Heterogeneous full/omit
        # share the identical source prefix; quality, not bit equality, compares hardware.
        with self.unit("cache-off-path"):
            common = self.generate(item, "off", role="a100", instrument=True)
            path = self.generate(item, "off", role="v100", action="full_migrate", instrument=True)
            quality = self.quality(path, common)
            self.record("correctness", "cache-off-path", status="passed" if known_quality(quality) else "failed",
                        continuous=common, full=path, quality=quality)
            if not known_quality(quality):
                raise RuntimeError("cache-off heterogeneous path failed common-reference quality")
        for role in self.workers:
            with self.unit("local-" + role):
                continuous = self.generate(item, "t008", role=role, instrument=True)
                full = self.generate(item, "t008", role=role, action="local_full_migrate", instrument=True)
                omit = None
                try:
                    omit = self.generate(item, "t008", role=role, action="local_certified_omit", instrument=True)
                except IneligibleAction:
                    pass
                exact = all(full[k] == continuous[k] for k in ("latent_sha256", "output_sha256", "traces"))
                if omit:
                    exact &= all(omit[k] == continuous[k] for k in ("latent_sha256", "output_sha256", "traces"))
                self.record("correctness", "local-" + role, status="passed" if exact else "failed",
                            continuous=continuous, full=full, certified_omit=omit, step_equivalent=exact)
                if not exact:
                    raise RuntimeError("same-device hot/omit step-by-step equivalence failed")
        with self.unit("state-correctness", estimate=self.job["unit_estimate_seconds"] * 4 / 3 + 30):
            reference = self.generate(item, "t008", role="v100", instrument=True)
            full = self.generate(item, "t008", role="v100", action="full_migrate", instrument=True)
            cold = self.generate(item, "t008", role="v100", action="cold_migrate", instrument=True)
            omitted = None
            try:
                omitted = self.generate(item, "t008", role="v100", action="certified_omit", instrument=True)
            except IneligibleAction as error:
                self.record("correctness", "certified-omit", status="ineligible", reason=str(error))
            exact = omitted is None or (full["latent_sha256"] == omitted["latent_sha256"]
                                       and full["output_sha256"] == omitted["output_sha256"])
            self.record("correctness", "migration", status="passed" if exact else "failed", continuous=reference,
                        full=full, cold=cold, certified_omit=omitted, exact_omit_equal=exact,
                        cross_device_quality=self.quality(full, reference),
                        instruction="cross-device numerical errors are empirical; exact omit must equal full on same path")
            if not exact:
                raise RuntimeError("certified omit differed from full migration")

    def calibrate(self):
        rows = []
        for role in self.workers:
            for item in inputs(self.config):
                case = role + "-" + item["prompt_id"] + "-" + str(item["seed"])
                with self.unit(case):
                    off = self.generate(item, role=role)
                    cached = self.generate(item, "t008", role=role)
                    row = self.record("calibration", case, status="completed", device_role=role, recipe_id="t008",
                                      prompt_id=item["prompt_id"], seed=item["seed"], split="dev",
                                      reference=off, candidate=cached, quality=self.quality(cached, off))
                    rows.append(row)
        sealed = [r for r in read_records(self.root, self.frozen) if r["type"] == "calibration"]
        gates = {role: cache_gate(sealed, self.config, role, self.context()) for role in self.workers}
        return self.write(self.job["phase"], dict(status="completed", cache_gates=gates))

    def migration(self):
        calibration = read_json(self.root / "calibrate.json")
        if not all(v["eligible"] for v in calibration["cache_gates"].values()):
            return self.write("migration", dict(status="direction_stopped", reason="cache development quality failed"))
        rows = []
        for item in [i for i in inputs(self.config) if i["prompt_id"] in ("kettle", "pendulum") and i["seed"] == 101]:
            with self.unit("screen-" + item["prompt_id"]):
                for action in ACTIONS:
                    try:
                        run = self.generate(item, "t008", role="v100", action=action)
                        row = self.record("migration_screen", action + "-" + item["prompt_id"],
                            **{k: run[k] for k in ("status", "prompt_id", "seed", "split", "elapsed_ms")},
                            action=action, run=run, performance_valid=True)
                    except IneligibleAction as error:
                        row = self.record("migration_screen", action + "-" + item["prompt_id"],
                            status="ineligible", prompt_id=item["prompt_id"], seed=item["seed"], split="dev",
                            action=action, reason=str(error))
                    rows.append(row)
        rows = [r for r in read_records(self.root, self.frozen) if r["type"] == "migration_screen"]
        selection = select_migration(rows, self.context())
        self.write("migration-selection", selection)
        if selection["selected"] is None:
            return self.write("migration", dict(status="unverified", selection=selection))
        gates = {}
        for split in ("dev", "holdout"):
            rows = []
            for item in inputs(self.config, split):
                case = split + "-" + item["prompt_id"] + "-" + str(item["seed"])
                with self.unit(case):
                    common = self.generate(item)
                    path = self.generate(item, role="v100", action="full_migrate")
                    path_quality = self.quality(path, common)
                    if not known_quality(path_quality):
                        self.record("migration_path_reference", case, status="failed", common_reference=common,
                                    path_reference=path, path_reference_quality=path_quality)
                        return self.write("migration", dict(status="quality_stopped", reason="cache-off path reference failed common reference",
                                                            selection=selection, gates=gates))
                    candidate = self.generate(item, "t008", role="v100", action=selection["selected"])
                    row = self.record("migration_quality", case, status="completed", action=selection["selected"],
                        split=split, prompt_id=item["prompt_id"], seed=item["seed"], common_reference=common,
                        path_reference=path, candidate=candidate, path_reference_quality=path_quality,
                        canonical_quality=self.quality(candidate, common), path_quality=self.quality(candidate, path))
                    rows.append(row)
            sealed = [r for r in read_records(self.root, self.frozen)
                      if r["type"] == "migration_quality" and r["split"] == split]
            gate = migration_quality_gate(sealed, self.config, split, selection["selected"], self.context())
            gates[split] = gate
            self.write("migration-" + split, gate)
            if not gate["eligible"]:
                return self.write("migration", dict(status="quality_stopped", selection=selection, gates=gates))
        return self.write("migration", dict(status="completed", selection=selection, gates=gates))

    def mechanism(self):
        migration = read_json(self.root / "migration.json")
        if migration["status"] != "completed":
            return self.write("mechanism", dict(status="unverified", reason="migration development and holdout required"))
        candidate = migration["selection"]["selected"]
        actions = list(dict.fromkeys(("stay", "full_migrate", candidate)))
        item = inputs(self.config)[0]
        screen = [r for r in read_records(self.root, self.frozen)
                  if r["type"] == "migration_screen" and r.get("performance_valid") and r.get("status") == "completed"]
        means = {a: statistics.mean(r["run"]["critical_path_ms"] for r in screen if r["action"] == a)
                 for a in ("stay", "full_migrate")}
        break_even = means["stay"] - means["full_migrate"]
        queues, queue_length = [], None
        # Queue probe runs real decoded requests and evaluates every output.
        for length in (1, 2, 4, 8) if break_even > 0 else ():
            with self.unit("queue-%d" % length):
                reference = self.generate(item)
                specs = [self.spec(item) for _ in range(length)]
                for spec in specs:
                    self.workers["a100"].call("warm", spec=spec)
                start = time.monotonic()
                runs = self.workers["a100"].call("batch", specs=specs)
                pending = [r for r in runs if r.get("artifacts_pending")]
                if pending:
                    saved = self.workers["a100"].call("flush", request_ids=[r["request_id"] for r in pending])
                    for run in pending:
                        run.update(saved[run["request_id"]])
                for run in runs:
                    run["identity"] = reference["identity"]
                    run["quality"] = self.quality(run, reference)
                observation = dict(queue_length=length, wait_ms=(runs[-1]["decoded_at_monotonic"] - start) * 1000,
                                   all_requests_decoded=True, performance_valid=True)
                self.record("queue_probe", str(length), status="completed", runs=runs, reference=reference,
                            quality_passed=all(known_quality(r["quality"]) for r in runs), **observation)
                queues.append(observation)
            queue_length = choose_queue(queues, break_even)
            if queue_length is not None:
                break
        trials = []
        for condition, length in [("idle", 0)] + ([("busy", queue_length)] if queue_length else []):
            for repetition, order in enumerate((actions, list(reversed(actions)))):
                with self.unit(condition + "-" + str(repetition)):
                    common = self.generate(item)
                    runs, qualities = {}, {}
                    for action in order:
                        run = self.generate(item, "t008", role="v100", action=action, queue=length)
                        runs[action] = run
                        qualities[action] = self.quality(run, common)
                        for queued in run["queued_outputs"]:
                            queued["identity"] = common["identity"]
                            queued["quality"] = self.quality(queued, common)
                    qualified = all(known_quality(q) for q in qualities.values()) and all(
                        known_quality(q["quality"]) for r in runs.values() for q in r["queued_outputs"])
                    coverage = condition == "idle" or all(r["target_wait_ms"] >= 1.25 * break_even
                                                          for a, r in runs.items() if a != "stay")
                    trial = self.record("mechanism", condition + "-" + str(repetition), status="completed",
                        condition=condition, order=order, queue_length=length, runs=runs, quality=qualities,
                        quality_passed=qualified, queue_coverage=coverage, all_requests_decoded=True, reference=common,
                        costs_ms={key: r["critical_path_ms"] for key, r in runs.items()},
                        simple_baseline="average_remaining_cost_plus_full_migration", performance_valid=True)
                    trials.append(trial)
        return self.write("mechanism", dict(status="completed", selected=candidate, queue_length=queue_length,
            busy_status="measured" if queue_length else "unverified_real_queue_did_not_cover_positive_break_even",
            break_even_wait_ms=break_even, unit_ids=[t["unit_id"] for t in trials]))

    def capacity(self):
        # The allocator is never padded. Each sweep rebuilds the same real paused inputs.
        calibration = read_json(self.root / "calibrate.json")
        if not calibration["cache_gates"]["a100"]["eligible"]:
            return self.write("capacity", dict(status="direction_stopped", reason="cache quality gate failed"))
        worker, grid = self.workers["a100"], inputs(self.config)
        reference_rows = [r for r in read_records(self.root, self.frozen) if r["type"] == "calibration" and r["device_role"] == "a100"]
        references = {(r["prompt_id"], r["seed"]): r["reference"] for r in reference_rows}
        trials = []
        for count in (1, 2, 4, 8):
            failed_keep = False
            for action in ("keep", "offload", "reclaim"):
                if action != "keep" and not failed_keep:
                    break
                with self.unit("%d-%s" % (count, action),
                               estimate=self.job["unit_estimate_seconds"] * (count + 1) / 2):
                    specs, completed, failed = [], [], False
                    try:
                        try:
                            for item in grid[:count]:
                                spec = self.spec(item, "t008")
                                worker.call("warm", spec=spec)
                                worker.call("prepare", spec=spec)
                                specs.append((spec, item))
                                worker.call("advance", request_id=spec["request_id"], until=16)
                        except RemoteError as error:
                            if error.remote_type != "OutOfMemoryError":
                                raise
                            self.record("capacity", "%d-%s-setup" % (count, action), status="setup_capacity_failure",
                                        reason=str(error), supported=False, failure_scope="paused_request_setup")
                            return self.write("capacity", dict(status="unverified", reason="paused request setup OOM is not admission evidence"))
                        before = worker.call("memory")
                        if action == "reclaim":
                            certs = [worker.call("certificate", request_id=s["request_id"]) for s, _ in specs]
                            if not all(c["eligible"] for c in certs):
                                raise IneligibleAction("at least one paused request lacks legal step16 certificate")
                        if action != "keep":
                            for s, _ in specs:
                                worker.call(action, request_id=s["request_id"])
                        try:
                            urgent = self.generate(grid[0], "off")
                        except RemoteError as error:
                            if error.remote_type != "OutOfMemoryError":
                                raise
                            failed = True
                            urgent = None
                        for s, item in specs:
                            if action == "offload":
                                worker.call("restore", request_id=s["request_id"])
                            worker.call("advance", request_id=s["request_id"], until=32)
                            run = worker.call("finish", request_id=s["request_id"])
                            run["identity"] = dict(context_sha256=self.context(), input=item, recipe="t008", action=action)
                            run["quality"] = self.quality(run, references[(item["prompt_id"], item["seed"])])
                            completed.append(run)
                        if urgent is not None:
                            urgent["quality"] = self.quality(urgent, references[(grid[0]["prompt_id"], grid[0]["seed"])])
                        quality = (not failed and all(known_quality(r["quality"]) for r in completed)
                                   and known_quality(urgent["quality"]))
                        trial = self.record("capacity", "%d-%s" % (count, action), status="completed",
                            paused_count=count, action=action, urgent_capacity_failed=failed, original_requests=completed,
                            urgent=urgent, before=before, quality_passed=quality, artificially_filled_memory=False,
                            supported=bool(failed_keep and quality and action != "keep"))
                        trials.append(trial)
                        failed_keep = failed_keep or (action == "keep" and failed)
                    except IneligibleAction as error:
                        self.record("capacity", "%d-%s" % (count, action), status="ineligible", reason=str(error))
                    finally:
                        for s, _ in specs:
                            with contextlib.suppress(Exception):
                                worker.call("release", request_id=s["request_id"])
            if failed_keep:
                break
        return self.write("capacity", dict(status="completed", supported=any(t["supported"] for t in trials),
                                           unit_ids=[t["unit_id"] for t in trials], max_paused_tested=count))


def run(job):
    experiment = DualExperiment(job)
    try:
        experiment.load()
        return getattr(experiment, job["phase"].split("-")[0])()
    finally:
        experiment.close()
